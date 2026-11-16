# -*- coding: utf-8 -*-
"""公共函数。所有算法默认参数在各自函数内；不使用全局配置。
坐标原点为图像左上角，x 向右，y 向下；四角顺序 TL/TR/BR/BL。
"""
from pathlib import Path
from collections import Counter
import csv
import json
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt





def save_trial_results(root, rows, char_ks, string_ks):
    """Save per-trial Top-K accuracies as percentages."""

    output_path = root / 'topk_per_trial.csv'

    fieldnames = ['Label', 'Experiment', 'Trial']
    fieldnames += [f'C@{k}' for k in char_ks]
    fieldnames += [f'S@{k}' for k in string_ks]

    with output_path.open(
        'w', newline='', encoding='utf-8-sig'
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n[CSV] Saved {len(rows)} trials to: {output_path}')

def read_positions(path):
    with path.open('r', encoding='utf-8-sig') as f:
        data = json.load(f)
    frames = data.get('frames', [])
    positions = []
    for frame in frames:
        ranking = frame.get('ranking', [])
        choices = []
        for original_index, entry in enumerate(ranking):
            distance = entry.get('distance')
            key = str(entry.get('key', ''))
            if (len(key) == 1 and key.isdigit() and
                    isinstance(distance, (int, float)) and
                    not isinstance(distance, bool) and math.isfinite(distance)):
                choices.append((key, float(distance), original_index))
        choices.sort(key=lambda item: item[1])
        if not choices:
            raise ValueError(f"frame {frame.get('frame_name')}: no valid digit choices")
        positions.append(choices)
    return positions


def evaluate_trial(positions, label, char_ks, string_ks):
    import heapq

    if len(positions) != len(label):
        raise ValueError(
            f'Position count ({len(positions)}) '
            f'does not match label length ({len(label)})'
        )

    char_ranks = []
    for choices, target in zip(positions, label):
        rank = next(
            (i + 1 for i, (key, _, _) in enumerate(choices)
             if key == target),
            math.inf
        )
        char_ranks.append(rank)

    # Generate only the Top-K combinations
    max_k = max(string_ks)
    initial = tuple(0 for _ in positions)

    initial_distance = sum(p[0][1] for p in positions)
    heap = [(initial_distance, initial)]
    visited = {initial}

    string_rank = math.inf

    for rank in range(1, max_k + 1):
        if not heap:
            break

        distance, indices = heapq.heappop(heap)

        word = ''.join(
            positions[i][j][0]
            for i, j in enumerate(indices)
        )

        if word == label:
            string_rank = rank
            break

        for i in range(len(positions)):
            if indices[i] + 1 >= len(positions[i]):
                continue

            next_indices = list(indices)
            next_indices[i] += 1
            next_indices = tuple(next_indices)

            if next_indices in visited:
                continue

            visited.add(next_indices)

            next_distance = (
                distance
                - positions[i][indices[i]][1]
                + positions[i][next_indices[i]][1]
            )

            heapq.heappush(
                heap,
                (next_distance, next_indices)
            )

    return (
        {k: sum(rank <= k for rank in char_ranks) for k in char_ks},
        {k: int(string_rank <= k) for k in string_ks}
    )


def original_stem(stem):
    DECIMAL_SUFFIX = re.compile(r'^\d+\.\d+$')
    """Strip consecutive trailing decimal metrics, e.g. _37.5_75.0."""
    parts = stem.split('_')
    while len(parts) > 1 and DECIMAL_SUFFIX.fullmatch(parts[-1]):
        parts.pop()
    return '_'.join(parts)



def read_json(path):
    with path.open('r', encoding='utf-8-sig') as f:
        return json.load(f)


def read_s4(path):
    """Read S4 events and their associated blue frames."""
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError('S4 contains no events')

    for row in rows:
        if not row.get('frame_name'):
            raise ValueError('S4 missing frame_name')

        if not row.get('all_frame_names'):
            raise ValueError('S4 missing all_frame_names')

        row['all_frame_names'] = [
            name.strip()
            for name in row['all_frame_names'].split(';')
            if name.strip()
        ]

        if row['frame_name'] not in row['all_frame_names']:
            raise ValueError(
                f"{row['frame_name']}: representative frame "
                f"not found in all_frame_names"
            )

    return rows


def get_ranking(frame):
    """
    Extract candidate ranking from the first ray.

    Consistent with the original S5 implementation.
    """
    rays = frame.get('rays') or []

    if not rays:
        raise ValueError(
            f"{frame['frame_name']}: rays is empty"
        )

    ranking = rays[0].get('ranking', [])

    if not ranking:
        raise ValueError(
            f"{frame['frame_name']}: ranking is empty"
        )

    return ranking


def get_distances(frame):
    """Extract key-distance mapping from the first ray."""
    ranking = get_ranking(frame)

    distances = {}

    for item in ranking:
        key = str(item['key'])
        distance = float(item['distance'])

        if not math.isfinite(distance) or distance < 0:
            raise ValueError(
                f"{frame['frame_name']}: invalid distance for {key}"
            )

        if key in distances:
            raise ValueError(
                f"{frame['frame_name']}: duplicate key {key}"
            )

        distances[key] = distance

    return distances


def calculate_score(key, event_distances):
    """
    Calculate RMS Distance for a candidate key.

        RMS = sqrt(mean(distance^2))

    Lower RMS indicates better geometric consistency
    with the event's rays.
    """
    distances = []

    for frame_name, frame_distances in event_distances:
        if key not in frame_distances:
            raise ValueError(
                f'{frame_name}: missing key {key}'
            )

        distances.append(frame_distances[key])

    return math.sqrt(
        sum(d ** 2 for d in distances) / len(distances)
    )


def filter_event(row, frames):
    """
    Process one event using RMS Distance.

    All candidate keys are evaluated across all blue frames.

    No original hit-based filtering.
    No forced zero distance.

    The best candidate is marked with hit=True.
    """
    representative_name = row['frame_name']
    event_names = row['all_frame_names']

    if representative_name not in frames:
        raise ValueError(
            f'Representative frame not found: {representative_name}'
        )

    # Collect distance mappings from all blue frames.
    event_distances = []

    for name in event_names:
        if name not in frames:
            raise ValueError(
                f'Event frame not found in S3: {name}'
            )

        distances = get_distances(frames[name])
        event_distances.append((name, distances))

    # Obtain all candidate keys from the first event frame.
    keys = list(event_distances[0][1].keys())

    # Ensure every frame contains the same candidate keys.
    expected_keys = set(keys)

    for name, distances in event_distances:
        if set(distances) != expected_keys:
            raise ValueError(
                f'{name}: inconsistent candidate keys'
            )

    # Calculate RMS Distance for every candidate.
    scores = {}

    for key in keys:
        scores[key] = calculate_score(
            key,
            event_distances
        )

    # Select the candidate with the minimum RMS Distance.
    best_key = min(
        keys,
        key=lambda key: (
            scores[key],
            key
        )
    )

    # Construct final ranking.
    # Keep the original key/distance/hit structure.
    output_ranking = [
        {
            'key': key,
            'distance': scores[key],
            'hit': key == best_key
        }
        for key in keys
    ]

    output_ranking.sort(
        key=lambda item: (
            item['distance'],
            item['key']
        )
    )

    # Keep representative frame name for compatibility.
    # Its distance values are not used in scoring.
    output_frame = {
        'frame_name': representative_name,
        'ranking': output_ranking
    }

    # Detailed event log.
    log = [
        f"Event {row.get('event_id', '?')}",
        f"Representative: {representative_name}",
        f"Frames: {event_names}",
        f"Frame count: {len(event_names)}",
        "RMS ranking:"
    ]

    for rank, item in enumerate(output_ranking, start=1):
        log.append(
            f"  {rank:2d}. "
            f"Key {item['key']}: "
            f"RMS = {item['distance']:.6f}"
        )

    log.extend([
        f"Selected: {best_key}",
        ""
    ])

    return output_frame, '\n'.join(log)


def process(s3_path, s4_path, dest):
    """Process all events in one experiment directory."""
    s3 = read_json(s3_path)
    rows = read_s4(s4_path)

    frames = {
        frame['frame_name']: frame
        for frame in s3.get('frames', [])
    }

    output_frames = []
    logs = []

    for row in rows:
        output_frame, log = filter_event(row, frames)

        output_frames.append(output_frame)
        logs.append(log)

    result = {
        'frames': output_frames
    }

    # Write files only after all events have been processed.
    dest.mkdir(parents=True, exist_ok=True)

    output_path = dest / 'S5_key_candidates_cleaned.json'

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8'
    )

    log_path = dest / 'S5_log.txt'

    log_path.write_text(
        '\n'.join(logs),
        encoding='utf-8'
    )

    return len(output_frames)

def clean_s3_mapping_debug(directory, events):
    """只保留 S4 选中帧对应的 S3 调试图片。"""
    debug_dir = directory / 'S3_mapping_debug'

    if not debug_dir.is_dir():
        print(f'S4: debug directory not found: {debug_dir}')
        return

    keep_frames = {event['frame_name'] for event in events}

    deleted = 0
    for image_path in debug_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            if image_path.name not in keep_frames:
                image_path.unlink()
                deleted += 1

    # print(f'S4: {debug_dir}: kept {len(keep_frames)} selected frames, deleted {deleted} images')

def select_ray_by_keyboard(tmp_str,ray, keyboard):
    """Discard non-intersecting rays and retain only the longest valid ray."""
    selected = keyboard.get('selected') if isinstance(keyboard, dict) else None
    corners = selected.get('corners') if isinstance(selected, dict) else None
    original = ray.get('candidates', [])

    if not corners or len(corners) < 3:
        ray['candidates'] = []
        ray['selected_index'] = None
        ray['blue_detected'] = False
        ray['ray_filter_status'] = 'keyboard_missing_or_unsupported'
        ray['ray_filter_counts'] = {'before': len(original), 'after': 0}
        return ray

    # 1. 保留与 keyboard 相交的 rays
    key_corners = get_key_area_corners(tmp_str,corners)
    kept = [c for c in original if ray_intersects_keyboard(c, key_corners)]

    # 2. 按长度降序排列
    kept.sort(key=lambda c: c.get('length_px', 0), reverse=True)

    # 3. 只保留最长的一条 ray
    ray['candidates'] = kept[:1]

    ray['selected_index'] = 0 if kept else None
    ray['blue_detected'] = bool(kept)
    ray['ray_filter_status'] = 'selected' if kept else 'no_ray_intersects_keyboard'
    ray['ray_filter_counts'] = {
        'before': len(original),
        'after': len(ray['candidates'])
    }

    return ray


def ray_intersects_keyboard(candidate, corners):
    """Test the forward half-line lower -> upper against the keyboard quadrilateral."""
    ox, oy = map(float, candidate['lower'])
    ux, uy = map(float, candidate['upper'])
    dx, dy = ux - ox, uy - oy
    if math.hypot(dx, dy) < 1e-9:
        return False
    polygon = [tuple(map(float, p)) for p in corners]
    if len(polygon) < 3:
        return False

    def cross(ax, ay, bx, by):
        return ax * by - ay * bx

    # Handle a ray starting inside the keyboard polygon.
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if (y1 > oy) != (y2 > oy):
            x_cross = x1 + (oy - y1) * (x2 - x1) / (y2 - y1)
            if ox < x_cross:
                inside = not inside
    if inside:
        return True

    # Ray/edge intersection: O + t D = A + u E, t>=0, 0<=u<=1.
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        ex, ey = bx - ax, by - ay
        qx, qy = ax - ox, ay - oy
        denom = cross(dx, dy, ex, ey)
        if abs(denom) < 1e-9:
            # Parallel and collinear: overlapping forward portion counts.
            if abs(cross(qx, qy, dx, dy)) < 1e-7:
                dd = dx * dx + dy * dy
                if max((qx * dx + qy * dy) / dd,
                       ((bx - ox) * dx + (by - oy) * dy) / dd) >= 0:
                    return True
            continue
        t = cross(qx, qy, ex, ey) / denom
        u = cross(qx, qy, dx, dy) / denom
        if t >= -1e-9 and -1e-9 <= u <= 1 + 1e-9:
            return True
    return False


def get_key_area_corners(tmp_str, corners):
    """根据外框四角，计算按键区域四角。"""

    # 固定的按键区域相对坐标
    x_min, x_max = 0.355, 0.667
    y_min, y_max = 0.364, 0.699

    if tmp_str in ["unlock4","unlock6",'unlock8']:
        # 新键盘（只包含 0–9 和 Del）
        x_min, x_max = 0.083, 0.907
        y_min, y_max = 0.385, 0.737
    if tmp_str in ["keypad4","keypad6",'keypad8']:
        # 新键盘（只包含 0–9 和 Del）
        x_min, x_max = 0.020, 0.980
        y_min, y_max = 0.213, 0.975

    tl, tr, br, bl = np.array(corners, dtype=float)

    def interpolate(x, y):
        return (
            (1 - x) * (1 - y) * tl
            + x * (1 - y) * tr
            + x * y * br
            + (1 - x) * y * bl
        ).tolist()

    return [
        interpolate(x_min, y_min),
        interpolate(x_max, y_min),
        interpolate(x_max, y_max),
        interpolate(x_min, y_max)
    ]



def extract_blue_events(detections, info, max_missing_frames=0):
    """Group consecutive blue detections in the time-sampled frame stream."""

    max_missing_frames = 0 if max_missing_frames is None else max_missing_frames
    if max_missing_frames < 0:
        raise ValueError('max_missing_frames must be >= 0')

    sampling_ms = float(info['interval_ms'])

    groups, active, pending = [], [], []
    previous_frame_id = None

    for frame in detections:
        frame_id = int(frame['frame_id'])

        # Check continuity in the sampled frame sequence
        if previous_frame_id is not None and frame_id - previous_frame_id != 1:
            if active:
                groups.append(active)
            active, pending = [], []

        previous_frame_id = frame_id

        blue = frame['blue_detected']
        if isinstance(blue, str):
            blue = blue.lower() == 'true'

        if blue:
            active.extend(pending)
            pending = []
            active.append(frame)

        elif active:
            pending.append(frame)
            if len(pending) > max_missing_frames:
                groups.append(active)
                active, pending = [], []

    if active:
        groups.append(active)

    events = []

    for group in groups:
        blue_frames = [
            f for f in group
            if f['blue_detected'] is True
            or str(f['blue_detected']).lower() == 'true'
        ]

        first, last = blue_frames[0], blue_frames[-1]

        # Middle observation
        mid = blue_frames[(len(blue_frames) - 1) // 2]

        start = float(first['timestamp_ms'])
        end = float(last['timestamp_ms'])

        representative = blue_frames[len(blue_frames) // 2]

        events.append({
            'event_id': len(events),
            'frame_name': representative['frame_name'],
            'all_frame_names': [f['frame_name'] for f in blue_frames],
            'start_ms': start,
            'end_ms': end,
            'observed_span_ms': end - start,
            'sampling_interval_ms': sampling_ms,
            'blue_frame_count': len(blue_frames),
            'first_frame_id': first['frame_id'],
            'last_frame_id': last['frame_id'],
            'representative_frame_id': mid['frame_id'],
            'status': 'segmented'
        })

    return events


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, records, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_image(path):
    # Windows 中文路径也能读取。
    frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f'Cannot read image: {path}')
    return frame


def save_image(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f'Cannot encode: {path}')
    data.tofile(str(path))


def discover_videos(input_root, selected_labels=None):
    """dataset_raw_split/<group>/<label>/**/<video>.mp4。保留整个相对路径。"""
    root = Path(input_root)
    extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}
    if not root.is_dir():
        raise FileNotFoundError(root)
    results = []
    for label_dir in sorted(root.glob('*/*')):
        if not label_dir.is_dir() or (selected_labels is not None and label_dir.name not in selected_labels):
            continue
        for video in sorted(label_dir.rglob('*')):
            if video.is_file() and video.suffix.lower() in extensions:
                relative = video.relative_to(root)
                # 文件扩展名也保留，避免 a.mp4 与 a.mov 的输出发生冲突。
                results.append({'video_path': video, 'label': label_dir.name,
                                'relative_path': relative, 'relative_output': relative.parent / video.name})
    return results


def video_directories(output_root):
    # print(output_root)
    # exit()
    directories = sorted(p.parent for p in Path(output_root).rglob('S1_video_info.json'))
    if not directories:
        raise RuntimeError(f'No extracted videos in {output_root}; run S1 first.')
    return directories



def reset_step_directory(video_dir, step_name):
    """清空指定步骤的旧结果，并重新创建目录。"""
    import shutil

    step_dir = Path(video_dir) / step_name

    if step_dir.exists():
        shutil.rmtree(step_dir)

    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir



def split_one_video(video_path, label, video_dir, interval_ms=100):
    """按视频时间戳抽帧，默认每100ms一帧，兼容VFR视频。"""

    if interval_ms <= 0:
        raise ValueError('interval_ms must be > 0')

    video_path, video_dir = Path(video_path), Path(video_dir)
    frames_dir = video_dir / 'S1_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f'Invalid FPS: {fps}')

    info = {
        'label': label,
        'source_video_path': str(video_path.resolve()),
        'fps': fps,
        'source_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'source_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'interval_ms': interval_ms,
        'timestamp_method': 'cv2.CAP_PROP_POS_MSEC'
    }

    records = []
    source_id = 0
    next_sample_ms = 0.0
    last_timestamp_ms = -1.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 读取当前帧的实际视频时间戳
            timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))

            if not np.isfinite(timestamp_ms) or timestamp_ms < 0:
                raise RuntimeError(
                    f'Invalid timestamp at source frame {source_id}: '
                    f'{timestamp_ms}'
                )

            if timestamp_ms + 1e-6 < last_timestamp_ms:
                raise RuntimeError(
                    f'Non-monotonic timestamps at source frame {source_id}'
                )

            last_timestamp_ms = timestamp_ms

            # 按时间间隔抽帧
            if timestamp_ms + 1e-6 >= next_sample_ms:

                name = f'frame_{len(records):05d}.jpg'
                save_image(frames_dir / name, frame)

                records.append({
                    'frame_id': len(records),
                    'source_frame_id': source_id,
                    'timestamp_ms': timestamp_ms,
                    'frame_name': name
                })

                # 跳到下一个尚未经过的采样时间点
                next_sample_ms = (
                    int((timestamp_ms + 1e-6) // interval_ms) + 1
                ) * interval_ms

            source_id += 1

    finally:
        cap.release()

    if not source_id:
        raise RuntimeError(f'No decoded frames: {video_path}')

    # 删除上次运行遗留的多余帧
    valid_names = {r['frame_name'] for r in records}
    for old_frame in frames_dir.glob('frame_*.jpg'):
        if old_frame.name not in valid_names:
            old_frame.unlink()

    info.update(
        source_frame_count=source_id,
        sampled_frame_count=len(records)
    )

    write_csv(
        video_dir / 'S1_frame_metadata.csv',
        records,
        ['frame_id', 'source_frame_id', 'timestamp_ms', 'frame_name']
    )

    write_json(video_dir / 'S1_video_info.json', info)

    print(
        f'S1: {video_path} -> {len(records)} frames '
        f'(interval={interval_ms}ms)'
    )


def debug_blue_rays(frame, ray_result):
    """在原始图像上显示最终保留的所有 ray。"""
    image = frame.copy()

    for i, ray in enumerate(ray_result['candidates']):
        lower = tuple(map(int, ray['lower']))
        upper = tuple(map(int, ray['upper']))

        # 绿色：普通候选；红色：selected ray
        color = (0, 0, 255) if i == ray_result['selected_index'] else (0, 255, 0)

        cv2.line(image, lower, upper, color, 3)
        cv2.circle(image, lower, 5, color, -1)
        cv2.circle(image, upper, 5, color, -1)

        cv2.putText(
            image,
            f"Ray {i}: {ray['length_px']:.1f}px",
            (upper[0] + 5, upper[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return image


def detect_blue_ray(frame, lower_blue=None, upper_blue=None):
    """HSV + 线段拟合。端点是可见线段端点，不等同于真实三维终点。
    返回所有合格候选；selected_index 是最长候选的启发式选择。
    """
    lower_blue = [90, 35, 35] if lower_blue is None else lower_blue
    upper_blue = [155, 255, 255] if upper_blue is None else upper_blue
    h, w = frame.shape[:2]
    min_length = max(25, 0.055 * h)
    min_elongation = 5.0
    min_support = 0.40
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_blue, np.uint8), np.array(upper_blue, np.uint8))


    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    thick_core = np.uint8(dist > 8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    large_region = cv2.dilate(
    thick_core,
    kernel,
    iterations=1
)
    thin_mask = cv2.bitwise_and(
        mask,
        cv2.bitwise_not(large_region)
    )

    # cv2.imshow("Blue Mask", thin_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()

    lines = cv2.HoughLinesP(thin_mask, 1, np.pi / 360, 18,
                            minLineLength=int(min_length), maxLineGap=12)

    # debug_hough = frame.copy()

    # if lines is not None:
    #     for x1, y1, x2, y2 in lines.reshape(-1, 4):
    #         cv2.line(
    #             debug_hough,
    #             (x1, y1),
    #             (x2, y2),
    #             (0, 255, 255),
    #             2
    #         )
    # cv2.imwrite("debug_hough.jpg", debug_hough)
    # exit()
    
    candidates = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            a, b = np.array([x1, y1], float), np.array([x2, y2], float)
            length = float(np.linalg.norm(b-a))
            if length < min_length or abs(y2-y1) < 0.65*length:
                continue
            # 当前视频控制器位于画面下方；蓝色 UI 常在画面上方。
            if max(y1, y2) < 0.80*h:
                continue
            unit = (b-a)/length
            normal = np.array([-unit[1], unit[0]])
            points = a + np.linspace(0, length, int(length)+1)[:, None]*unit
            support = []
            widths = []
            for point in points:
                offsets = np.arange(-8, 9)
                xy = np.rint(point + offsets[:, None]*normal).astype(int)
                valid = (xy[:, 0] >= 0)&(xy[:, 0] < w)&(xy[:, 1] >= 0)&(xy[:, 1] < h)
                values = mask[xy[valid, 1], xy[valid, 0]] > 0
                support.append(bool(values.any()))
                widths.append(int(values.sum()))
            score = float(np.mean(support))
            thickness = max(1.0, float(np.median(widths)))
            if score < min_support or length/thickness < min_elongation:
                continue
            # 仅按 y 排序为 lower/upper，明确不是三维传播方向。
            lower, upper = (a, b) if a[1] >= b[1] else (b, a)
            if any(np.linalg.norm(upper-np.array(c['upper'])) < 16 and
                   np.linalg.norm(lower-np.array(c['lower'])) < 30 for c in candidates):
                continue
            candidates.append({'lower': lower.tolist(), 'upper': upper.tolist(),
                               'length_px': length, 'color_support': score})
    candidates.sort(key=lambda c: c['length_px'], reverse=True)
    merged = []
    for candidate in candidates:
        a, b = np.array(candidate['upper']), np.array(candidate['lower'])
        duplicate = False
        for kept in merged:
            ka, kb = np.array(kept['upper']), np.array(kept['lower'])
            unit = (kb-ka)/np.linalg.norm(kb-ka)
            normal = np.array([-unit[1], unit[0]])
            if max(abs(np.dot(a-ka, normal)), abs(np.dot(b-ka, normal))) < 8:
                projections = [np.dot(a-ka, unit), np.dot(b-ka, unit)]
                if min(projections) <= kept['length_px']+12 and max(projections) >= -12:
                    duplicate = True
                    break
        if not duplicate:
            merged.append(candidate)
    candidates = merged
    return {'blue_detected': bool(candidates), 'candidates': candidates,
            'selected_index': 0 if candidates else None,
            'parameters': {'lower_blue': lower_blue, 'upper_blue': upper_blue,
                           'min_length_px': min_length, 'min_support': min_support}}




def detect_blue_ray_2(frame, lower_blue=None, upper_blue=None):  # 修改之前的
    """HSV + 线段拟合。端点是可见线段端点，不等同于真实三维终点。
    返回所有合格候选；selected_index 是最长候选的启发式选择。
    """
    lower_blue = [90, 35, 35] if lower_blue is None else lower_blue
    upper_blue = [155, 255, 255] if upper_blue is None else upper_blue
    h, w = frame.shape[:2]
    min_length = max(25, 0.055 * h)
    min_elongation = 5.0
    min_support = 0.40
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_blue, np.uint8), np.array(upper_blue, np.uint8))


    # cv2.imshow("Blue Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()

    lines = cv2.HoughLinesP(mask, 1, np.pi / 360, 18,
                            minLineLength=int(min_length), maxLineGap=12)

    # debug_hough = frame.copy()

    # if lines is not None:
    #     for x1, y1, x2, y2 in lines.reshape(-1, 4):
    #         cv2.line(
    #             debug_hough,
    #             (x1, y1),
    #             (x2, y2),
    #             (0, 255, 255),
    #             2
    #         )
    # cv2.imwrite("debug_hough.jpg", debug_hough)
    # exit()
    
    candidates = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            a, b = np.array([x1, y1], float), np.array([x2, y2], float)
            length = float(np.linalg.norm(b-a))
            if length < min_length or abs(y2-y1) < 0.65*length:
                continue
            # 当前视频控制器位于画面下方；蓝色 UI 常在画面上方。
            if max(y1, y2) < 0.80*h:
                continue
            unit = (b-a)/length
            normal = np.array([-unit[1], unit[0]])
            points = a + np.linspace(0, length, int(length)+1)[:, None]*unit
            support = []
            widths = []
            for point in points:
                offsets = np.arange(-8, 9)
                xy = np.rint(point + offsets[:, None]*normal).astype(int)
                valid = (xy[:, 0] >= 0)&(xy[:, 0] < w)&(xy[:, 1] >= 0)&(xy[:, 1] < h)
                values = mask[xy[valid, 1], xy[valid, 0]] > 0
                support.append(bool(values.any()))
                widths.append(int(values.sum()))
            score = float(np.mean(support))
            thickness = max(1.0, float(np.median(widths)))
            if score < min_support or length/thickness < min_elongation:
                continue
            # 仅按 y 排序为 lower/upper，明确不是三维传播方向。
            lower, upper = (a, b) if a[1] >= b[1] else (b, a)
            if any(np.linalg.norm(upper-np.array(c['upper'])) < 16 and
                   np.linalg.norm(lower-np.array(c['lower'])) < 30 for c in candidates):
                continue
            candidates.append({'lower': lower.tolist(), 'upper': upper.tolist(),
                               'length_px': length, 'color_support': score})
    candidates.sort(key=lambda c: c['length_px'], reverse=True)
    merged = []
    for candidate in candidates:
        a, b = np.array(candidate['upper']), np.array(candidate['lower'])
        duplicate = False
        for kept in merged:
            ka, kb = np.array(kept['upper']), np.array(kept['lower'])
            unit = (kb-ka)/np.linalg.norm(kb-ka)
            normal = np.array([-unit[1], unit[0]])
            if max(abs(np.dot(a-ka, normal)), abs(np.dot(b-ka, normal))) < 8:
                projections = [np.dot(a-ka, unit), np.dot(b-ka, unit)]
                if min(projections) <= kept['length_px']+12 and max(projections) >= -12:
                    duplicate = True
                    break
        if not duplicate:
            merged.append(candidate)
    candidates = merged
    return {'blue_detected': bool(candidates), 'candidates': candidates,
            'selected_index': 0 if candidates else None,
            'parameters': {'lower_blue': lower_blue, 'upper_blue': upper_blue,
                           'min_length_px': min_length, 'min_support': min_support}}


def order_corners(points):
    points = np.asarray(points, np.float32).reshape(4, 2)
    # 极角排序，再从最靠左上角的顶点开始；适用于当前近正视平面。
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0])
    points = points[np.argsort(angles)]
    return np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)

# -----------------------------------------------

import cv2
import numpy as np


def order_corners(points):
    """Order four points TL, TR, BR, BL."""
    p = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)],
                     p[np.argmax(s)], p[np.argmax(d)]], dtype=np.float32)


def largest_inscribed_rectangle(mask, min_aspect=1.1, max_aspect=3.5,
                                 min_area=0, max_area=None):
    """Largest all-white axis-aligned rectangle with aspect and area constraints.

    Returns (x1, y1, x2, y2, area), with exclusive right/bottom bounds.
    O(height * width * log(width)) worst-case with aspect-constrained candidates.
    """
    h, w = mask.shape
    if max_area is None:
        max_area = h * w
    heights = np.zeros(w, dtype=np.int32)
    best = None
    best_area = 0
    for y in range(h):
        heights = np.where(mask[y] > 0, heights + 1, 0)
        stack = []
        for x in range(w + 1):
            current = int(heights[x]) if x < w else 0
            while stack and heights[stack[-1]] > current:
                idx = stack.pop()
                max_h = int(heights[idx])
                left = stack[-1] + 1 if stack else 0
                width = x - left
                # Candidate heights for a fixed width; tallest feasible wins.
                lo = max(1, int(np.ceil(width / max_aspect)),
                         int(np.ceil(min_area / width)))
                hi = min(max_h, int(np.floor(width / min_aspect)),
                         int(np.floor(max_area / width)))
                if hi >= lo and width * hi > best_area:
                    best_area = width * hi
                    best = (left, y - hi + 1, x, y + 1, best_area)
            stack.append(x)
    return best


def _line_from_points(points):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0], float), np.array([vx, vy], float)


def _intersection(line_a, line_b):
    if line_a is None or line_b is None:
        return None
    p, d = line_a
    q, e = line_b
    det = d[0] * e[1] - d[1] * e[0]
    if abs(det) < 1e-5:
        return None
    v = q - p
    t = (v[0] * e[1] - v[1] * e[0]) / det
    return p + t * d


def expand_rectangle_to_quad(mask, rect, search_ratio=0.22,
                             min_edge_points=12):
    """Fit four boundaries by scanning outward from an inscribed rectangle.

    Returns quad (TL,TR,BR,BL) or None. Assumes the desired object's boundary
    remains the first white-to-black transition in each scan direction.
    """
    x1, y1, x2, y2, _ = rect
    h, w = mask.shape
    rw, rh = x2 - x1, y2 - y1
    max_dx = max(3, int(rw * search_ratio))
    max_dy = max(3, int(rh * search_ratio))
    # Sample away from corners, where neighboring shapes can cause interference.
    xs = np.unique(np.linspace(x1 + 0.10 * rw, x2 - 0.10 * rw, 45).astype(int))
    ys = np.unique(np.linspace(y1 + 0.10 * rh, y2 - 0.10 * rh, 35).astype(int))
    edges = {'top': [], 'bottom': [], 'left': [], 'right': []}

    for x in xs:
        x = int(np.clip(x, 0, w - 1))
        # Scan to first black pixel; last white pixel is boundary estimate.
        yy = max(0, y1 - max_dy)
        for y in range(y1, yy - 1, -1):
            if mask[y, x] == 0:
                if y < y1:
                    edges['top'].append((x, y + 1))
                break
        yy = min(h - 1, y2 - 1 + max_dy)
        for y in range(y2 - 1, yy + 1):
            if mask[y, x] == 0:
                if y >= y2:
                    edges['bottom'].append((x, y - 1))
                break

    for y in ys:
        y = int(np.clip(y, 0, h - 1))
        xx = max(0, x1 - max_dx)
        for x in range(x1, xx - 1, -1):
            if mask[y, x] == 0:
                if x < x1:
                    edges['left'].append((x + 1, y))
                break
        xx = min(w - 1, x2 - 1 + max_dx)
        for x in range(x2 - 1, xx + 1):
            if mask[y, x] == 0:
                if x >= x2:
                    edges['right'].append((x - 1, y))
                break

    if any(len(points) < min_edge_points for points in edges.values()):
        return None
    lines = {name: _line_from_points(points) for name, points in edges.items()}
    pairs = [('top', 'left'), ('top', 'right'),
             ('bottom', 'right'), ('bottom', 'left')]
    quad = [_intersection(lines[a], lines[b]) for a, b in pairs]
    if any(p is None for p in quad):
        return None
    quad = np.asarray(quad, dtype=np.float32)
    if not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
        return None
    area = abs(cv2.contourArea(quad))
    if area < 0.9 * rect[4] or area > 2.0 * rect[4]:
        return None
    if np.any(quad[:, 0] < 0) or np.any(quad[:, 0] >= w) or np.any(quad[:, 1] < 0) or np.any(quad[:, 1] >= h):
        return None
    return quad



# def detect_keyboard(tmp_str, frame, debug_path=None):
#     """Two-stage keyboard detection: interior rectangle, then boundary quad."""
#     black_threshold = 25
#     min_area_ratio, max_area_ratio = 0.035, 0.70
#     min_aspect, max_aspect = 1.1, 3.5
#     if tmp_str == 'unlock':
#         min_aspect=0.55
#         max_aspect=0.80

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     black_mask = cv2.inRange(gray, 0, black_threshold)
#     black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN,
#                                  np.ones((15, 15), np.uint8))
#     black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE,
#                                  np.ones((5, 5), np.uint8))
#     white_mask = black_mask

#     # # 保存调试图片
#     # cv2.imwrite("debug_01_gray.png", gray)
#     # cv2.imwrite("debug_02_threshold.png", black_mask)
#     # cv2.imwrite("debug_03_open.png", opened_mask)
#     # cv2.imwrite("debug_04_white_mask.png", white_mask)
#     # exit()

#     rect = largest_inscribed_rectangle(
#         white_mask, min_aspect=min_aspect, max_aspect=max_aspect,
#         min_area=int(min_area_ratio * gray.size),
#         max_area=int(max_area_ratio * gray.size))


#     # # ============================
#     # # Debug: largest rectangle
#     # # ============================

#     # vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)

#     # if rect is not None:
#     #     x1, y1, x2, y2, area = rect

#     #     width = x2 - x1
#     #     height = y2 - y1

#     #     print("===== Rectangle Detection =====")
#     #     print(f"Rectangle: {rect}")
#     #     print(f"Width: {width}")
#     #     print(f"Height: {height}")
#     #     print(f"W/H: {width / height:.3f}")
#     #     print(f"H/W: {height / width:.3f}")
#     #     print(f"Area ratio: {area / gray.size:.4f}")

#     #     cv2.rectangle(
#     #         vis,
#     #         (x1, y1),
#     #         (x2 - 1, y2 - 1),
#     #         (0, 0, 255),
#     #         3
#     #     )

#     # else:
#     #     print("No rectangle detected!")

#     # cv2.imwrite("debug_largest_rectangle.png", vis)
#     # exit()
    
#     candidates = []
#     quad = expand_rectangle_to_quad(white_mask, rect) if rect else None



#     vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)

#     if quad is not None:
#         cv2.polylines(
#             vis,
#             [np.rint(quad).astype(np.int32)],
#             True,
#             (0, 255, 0),
#             2
#         )

#         print("Detected quad:")
#         print(quad)

#     plt.figure(figsize=(8, 10))
#     plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()
#     print("----", rect)
#     exit()

#     if quad is not None:
#         area = float(abs(cv2.contourArea(quad)))
#         sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
#         aspect = float((sides[0] + sides[2]) / max(sides[1] + sides[3], 1e-9))
#         if min_area_ratio * gray.size <= area <= max_area_ratio * gray.size and min_aspect <= aspect <= max_aspect:
#             candidates.append({'corners': quad.tolist(), 'area_px': area,
#                                'fill_ratio': float(rect[4] / area)})
#     debug_path = "111.jpg"
#     if debug_path is not None:
#         vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
#         if rect:
#             x1, y1, x2, y2, _ = rect
#             cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (0, 200, 255), 2)
#         if quad is not None:
#             cv2.polylines(vis, [np.rint(quad).astype(np.int32)], True, (0, 255, 0), 2)
#         cv2.imwrite(str(debug_path), vis)
#     return {'selected': candidates[0] if candidates else None,
#             'candidates': candidates,
#             'parameters': {'black_threshold': black_threshold,
#                            'min_area_ratio': min_area_ratio,
#                            'max_area_ratio': max_area_ratio}}





def detect_keyboard(tmp_str, frame):
    """Two-stage keyboard detection: interior rectangle, then boundary quad."""
    black_threshold = 25
    min_area_ratio, max_area_ratio = 0.035, 0.8
    min_aspect, max_aspect = 1.1, 3.5
    if tmp_str in ["unlock4","unlock6",'unlock8']:
        min_aspect=0.55
        max_aspect=0.80
    if tmp_str in ["keypad4","keypad6",'keypad8']:
        min_aspect=0.90
        max_aspect=1.10
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black_mask = cv2.inRange(gray, 0, black_threshold)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN,
                                 np.ones((15, 15), np.uint8))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE,
                                 np.ones((5, 5), np.uint8))
    white_mask = black_mask

    if 0:
        vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        # cv2.imwrite(str(debug_path), vis)
        plt.figure(figsize=(8, 10))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        exit()

    rect = largest_inscribed_rectangle(
        white_mask, min_aspect=min_aspect, max_aspect=max_aspect,
        min_area=int(min_area_ratio * gray.size),
        max_area=int(max_area_ratio * gray.size))

    if 0:
        vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        if rect is not None:
            x1, y1, x2, y2, area = rect
            width = x2 - x1
            height = y2 - y1
            cv2.rectangle(
                vis,
                (x1, y1),
                (x2 - 1, y2 - 1),
                (0, 0, 255),
                3
            )
        plt.figure(figsize=(8, 10))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        exit()
            
    candidates = []
    quad = expand_rectangle_to_quad(white_mask, rect) if rect else None

    if 0:
        vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        if quad is not None:
            cv2.polylines(
                vis,
                [np.rint(quad).astype(np.int32)],
                True,
                (0, 255, 0),
                2
            )
        plt.figure(figsize=(8, 10))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        exit()

    if quad is not None:
        area = float(abs(cv2.contourArea(quad)))
        sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        aspect = float((sides[0] + sides[2]) / max(sides[1] + sides[3], 1e-9))
        if min_area_ratio * gray.size <= area <= max_area_ratio * gray.size and min_aspect <= aspect <= max_aspect:
            candidates.append({'corners': quad.tolist(), 'area_px': area,
                               'fill_ratio': float(rect[4] / area)})
    if 0:
        vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        if rect:
            x1, y1, x2, y2, _ = rect
            cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (0, 200, 255), 2)
        if quad is not None:
            cv2.polylines(vis, [np.rint(quad).astype(np.int32)], True, (0, 255, 0), 2)
        # cv2.imwrite(str(debug_path), vis)
        plt.figure(figsize=(8, 10))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        exit()

    return {'selected': candidates[0] if candidates else None,
            'candidates': candidates,
            'parameters': {'black_threshold': black_threshold,
                           'min_area_ratio': min_area_ratio,
                           'max_area_ratio': max_area_ratio}}



# def detect_keyboard(frame):
#     """暗色连通轮廓四边形检测，无法凭黑框推断键盘真实布局。"""
#     black_threshold = 25
#     min_area_ratio, max_area_ratio = 0.035, 0.70
#     min_fill = 0.82
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     mask = cv2.inRange(gray, 0, black_threshold)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    
#     cv2.imshow("Blue Mask", mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     exit()
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     candidates = []
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if not min_area_ratio*gray.size <= area <= max_area_ratio*gray.size:
#             continue
#         polygon = cv2.approxPolyDP(contour, 0.018*cv2.arcLength(contour, True), True)
#         if len(polygon) != 4 or not cv2.isContourConvex(polygon):
#             continue
#         corners = order_corners(polygon)
#         sides = np.linalg.norm(np.roll(corners, -1, axis=0)-corners, axis=1)
#         aspect = (sides[0]+sides[2])/(sides[1]+sides[3])
#         fill = area/max(cv2.contourArea(corners), 1)
#         if 1.1 <= aspect <= 3.5 and fill >= min_fill:
#             candidates.append({'corners': corners.tolist(), 'area_px': float(area),
#                                'fill_ratio': float(fill)})
#     candidates.sort(key=lambda c: c['area_px'], reverse=True)
#     return {'selected': candidates[0] if candidates else None, 'candidates': candidates,
#             'parameters': {'black_threshold': black_threshold, 'min_area_ratio': min_area_ratio,
#                            'max_area_ratio': max_area_ratio}}


def keyboard_layout_template():
    """示意 QWERTY 模板，必须用真实 UI 标定，不能直接用于实验结论。"""
    keys = []
    for row_id, row in enumerate(['1234567890', 'QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM']):
        x_offset = [0, 0, .035, .105][row_id]
        for col, name in enumerate(row):
            x, y = .035+x_offset+col*.09, .08+row_id*.18
            keys.append({'key': name, 'rect': [x, y, .080, .15]})
    keys.append({'key': 'SPACE', 'rect': [.25, .82, .45, .12]})
    return {'calibrated': False, 'note': 'Replace with actual keyboard layout; normalized panel coordinates.',
            'keys': keys}


def map_keyboard(corners, layout):
    source = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    matrix = cv2.getPerspectiveTransform(source, np.float32(corners))
    keys = []
    for item in layout['keys']:
        x, y, w, h = item['rect']
        if not (w > 0 and h > 0 and x >= 0 and y >= 0 and x+w <= 1.00001 and y+h <= 1.00001):
            raise ValueError(f'Invalid normalized rectangle: {item}')
        polygon = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
        projected = cv2.perspectiveTransform(polygon[None], matrix)[0]
        keys.append({'key': item['key'], 'polygon': projected.tolist()})
    return keys


def identify_key(ray, keys, max_endpoint_extension=None):
    """默认只取可见 upper 端点。可选少量延长仍只是推断，不是三维交点。"""
    max_endpoint_extension = 0 if max_endpoint_extension is None else max_endpoint_extension
    point = np.array(ray['upper'], float)
    direction = point - np.array(ray['lower'], float)
    direction /= max(np.linalg.norm(direction), 1e-9)
    for distance in np.arange(0, max_endpoint_extension+0.1, 1):
        candidate = point + distance*direction
        hits = [k['key'] for k in keys if cv2.pointPolygonTest(
            np.float32(k['polygon']), tuple(candidate), False) >= 0]
        if len(hits) == 1:
            return hits[0], candidate.tolist(), 'visible_endpoint' if distance == 0 else 'extrapolated_endpoint'
        if len(hits) > 1:
            return None, candidate.tolist(), 'ambiguous_overlap'
    return None, point.tolist(), 'endpoint_outside_keys'


def draw_detection(frame, ray, keyboard, keys=None, hit=None):
    image = frame.copy()
    for candidate in ray['candidates']:
        cv2.line(image, tuple(map(int, candidate['lower'])), tuple(map(int, candidate['upper'])), (255, 255, 0), 2)
    if keyboard['selected']:
        cv2.polylines(image, [np.int32(keyboard['selected']['corners'])], True, (0, 255, 0), 2)
    for item in keys or []:
        poly = np.int32(item['polygon'])
        cv2.polylines(image, [poly], True, (0, 255, 255), 1)
        center = poly.mean(axis=0).astype(int)
        cv2.putText(image, item['key'], tuple(center), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
    if hit is not None:
        cv2.circle(image, tuple(map(int, hit)), 5, (0, 0, 255), -1)
    return image



# === S3: labeled test-grid projection geometry ===
def s3_homography(frame_corners, layout):
    """Image coordinates -> reference-photo coordinates (planar projective mapping)."""
    src = np.asarray(frame_corners, np.float32).reshape(4, 2)
    dst = np.asarray(layout['reference_corners'], np.float32).reshape(4, 2)
    if abs(cv2.contourArea(src)) < 20 or abs(cv2.contourArea(dst)) < 20:
        raise ValueError('Degenerate panel corners')
    H = cv2.getPerspectiveTransform(src, dst)
    if not np.isfinite(H).all() or abs(float(np.linalg.det(H))) < 1e-12:
        raise ValueError('Invalid homography')
    return H


def s3_transform(points, H):
    a = np.asarray(points, np.float32).reshape(-1, 1, 2)
    result = cv2.perspectiveTransform(a, H).reshape(-1, 2)
    if not np.isfinite(result).all():
        raise ValueError('Invalid projected coordinates')
    return result


def s3_center_line_distance(center, a, b):
    """Perpendicular distance to an infinite 2D line, plus projection foot."""
    c, a, b = [np.asarray(v, dtype=float) for v in (center, a, b)]
    v = b-a
    length2 = float(v @ v)
    if length2 < 1e-10:
        raise ValueError('Degenerate ray segment')
    t = float((c-a) @ v / length2)
    foot = a + t*v
    return float(np.linalg.norm(c-foot)), foot.tolist()


def s3_rank_candidates(ray, H, layout):
    """Rank all test-grid cells. Threshold = 1/2 of each cell's short side."""
    lower, upper = s3_transform([ray['lower'], ray['upper']], H)
    result = {'ray_local': {'lower': lower.tolist(), 'upper': upper.tolist()},
              'ranking': [], 'candidates': [], 'status': 'ok'}
    if np.linalg.norm(upper-lower) < 1e-5:
        result['status'] = 'degenerate_ray'
        return result
    for item in layout['keys']:
        x, y, w, h = map(float, item['rect'])
        if min(w, h) <= 0:
            raise ValueError(f'Invalid cell rectangle: {item}')
        center = [x+w/2, y+h/2]
        d, foot = s3_center_line_distance(center, lower, upper)
        threshold = min(w, h)/2
        result['ranking'].append({
            'key': str(item['key']), 'center': [round(n, 3) for n in center],
            'foot': [round(n, 3) for n in foot],
            'distance': round(d, 4), 'threshold': round(threshold, 4),
            'hit': bool(d <= threshold)})
    result['ranking'].sort(key=lambda r: (r['distance'], r['key']))
    result['candidates'] = [r for r in result['ranking'] if r['hit']]
    result['status'] = ('no_candidates' if not result['candidates'] else
                        'single_candidate' if len(result['candidates']) == 1 else
                        'ambiguous_candidates')
    return result


def s3_project_keys(frame_corners, layout):
    H = s3_homography(frame_corners, layout)
    inv = np.linalg.inv(H)
    projected = []
    for item in layout['keys']:
        x, y, w, h = map(float, item['rect'])
        poly = s3_transform([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], inv)
        projected.append({'key': str(item['key']), 'polygon': poly.tolist()})
    return projected


def s3_clip_line_to_rect(p0, p1, rect):
    """Clip infinite 2D line to axis-aligned rect (xmin,ymin,xmax,ymax)."""
    a, b = np.asarray(p0, float), np.asarray(p1, float)
    v = b-a
    xmin,ymin,xmax,ymax = map(float,rect)
    hits=[]
    for x in (xmin,xmax):
        if abs(v[0]) > 1e-9:
            t=(x-a[0])/v[0]; y=a[1]+t*v[1]
            if ymin-1e-6 <= y <= ymax+1e-6: hits.append([x,y])
    for y in (ymin,ymax):
        if abs(v[1]) > 1e-9:
            t=(y-a[1])/v[1]; x=a[0]+t*v[0]
            if xmin-1e-6 <= x <= xmax+1e-6: hits.append([x,y])
    if len(hits)<2: return None
    import itertools
    return max(itertools.combinations(hits,2),key=lambda q:float(np.linalg.norm(np.subtract(*q))))


def s3_debug_pair(frame, panel_corners, ray, layout, result):
    """Left: source frame + extended ray. Right: reference panel + distances."""
    img = frame.copy()
    height,width=img.shape[:2]
    green=(42,190,90); gray=(155,155,155); yellow=(0,220,255)
    H=s3_homography(panel_corners,layout)
    inv=np.linalg.inv(H)
    cv2.polylines(img,[np.rint(panel_corners).astype(np.int32)],True,yellow,2)
    status={item['key']:item for item in result.get('ranking',[])}
    for item in s3_project_keys(panel_corners,layout):
        poly=np.rint(item['polygon']).astype(np.int32)
        color=green if status.get(item['key'],{}).get('hit') else gray
        cv2.polylines(img,[poly],True,color,2)
        c=poly.mean(axis=0).astype(int)
        cv2.putText(img,item['key'],tuple(c),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    visible_a=np.asarray(ray['lower'],float); visible_b=np.asarray(ray['upper'],float)
    cv2.line(img,tuple(np.rint(visible_a).astype(int)),tuple(np.rint(visible_b).astype(int)),(255,140,0),3)
    bounds=s3_clip_line_to_rect(visible_a,visible_b,(0,0,width-1,height-1))
    if bounds:
        cv2.line(img,*[tuple(np.rint(p).astype(int)) for p in bounds],yellow,2,cv2.LINE_AA)
    # Debug-only viewport: crop to the labeled grid rather than the full Store panel.
    # This changes only the rendering, not H, distances, thresholds, or JSON output.
    cells = np.array([list(map(float, item['rect'])) for item in layout['keys']])
    xmin = int(np.floor(np.min(cells[:, 0]) - 25))
    ymin = int(np.floor(np.min(cells[:, 1]) - 55))  # title space
    xmax = int(np.ceil(np.max(cells[:, 0] + cells[:, 2]) + 25))
    ymax = int(np.ceil(np.max(cells[:, 1] + cells[:, 3]) + 25))
    W, Hh = xmax - xmin, ymax - ymin
    canvas = np.full((Hh, W, 3), 248, np.uint8)
    offset = np.array([xmin, ymin], float)
    normray = result.get('ray_local', {})
    if 'lower' in normray and 'upper' in normray:
        a = np.asarray(normray['lower'], float) - offset
        b = np.asarray(normray['upper'], float) - offset
        clipped = s3_clip_line_to_rect(a, b, (0, 0, W-1, Hh-1))
        if clipped:
            cv2.line(canvas, *[tuple(np.rint(p).astype(int)) for p in clipped],
                     (0, 150, 240), 2, cv2.LINE_AA)
    for item in layout['keys']:
        name = str(item['key'])
        x, y, w, h = map(float, item['rect'])
        p1 = np.rint(np.array([x, y]) - offset).astype(int)
        p2 = np.rint(np.array([x+w, y+h]) - offset).astype(int)
        info = status.get(name)
        color = green if info and info['hit'] else gray
        cv2.rectangle(canvas, tuple(p1), tuple(p2), color, 2)
        cx, cy = np.rint(np.array([x+w/2, y+h/2]) - offset).astype(int)
        cv2.circle(canvas, (cx, cy), 4, color, -1)
        if info:
            foot = np.rint(np.asarray(info['foot']) - offset).astype(int)
            cv2.line(canvas, (cx, cy), tuple(foot), color, 1, cv2.LINE_AA)
            label = f'{name}: {info["distance"]:.1f}'
        else:
            label = name
        cv2.putText(canvas, label, (p1[0]+3, cy-8),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Normalized 2D keyboard area', (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 2, cv2.LINE_AA)
    # Fill 94% of the source frame's height. Upscaling is intentional here.
    scale = 0.94 * height / max(Hh, 1)
    right = cv2.resize(canvas, (max(1, round(W*scale)), max(1, round(Hh*scale))),
                       interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    padded = np.full((height, right.shape[1], 3), 250, np.uint8)
    y0 = (height-right.shape[0]) // 2
    padded[y0:y0+right.shape[0], :] = right
    cv2.putText(img,'Original frame + extrapolated 2D line',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,yellow,2)
    return np.hstack([img,padded])


def extract_key_events(rows, info, max_missing_frames=None):
    """连续蓝色段 -> 候选确认事件。默认不跨越任何非蓝色帧。
    unknown 与 missing 也保留；重复同键但中间变白仍为两个事件。
    """
    max_missing_frames = 0 if max_missing_frames is None else max_missing_frames
    if max_missing_frames < 0:
        raise ValueError('max_missing_frames must be >= 0')

    sampling_ms = float(info['interval_ms'])

    groups, current, pending = [], [], []
    previous = None

    for row in rows:
        frame_id = int(row['frame_id'])

        # 使用抽帧后的连续编号，而不是原始视频帧编号
        if previous is not None and frame_id - previous != 1:
            if current:
                groups.append(current)
            current, pending = [], []

        previous = frame_id

        if str(row['blue_detected']).lower() == 'true':
            current.extend(pending)
            pending = []
            current.append(row)

        elif current:
            pending.append(row)
            if len(pending) > max_missing_frames:
                groups.append(current)
                current, pending = [], []

    if current:
        groups.append(current)

    events = []

    for group in groups:
        blue = [
            r for r in group
            if str(r['blue_detected']).lower() == 'true'
        ]

        votes = Counter(r['key'] for r in blue if r['key'])
        ranking = votes.most_common()

        key = (
            ranking[0][0]
            if ranking and (
                len(ranking) == 1
                or ranking[0][1] > ranking[1][1]
            )
            else ''
        )

        support = votes[key] / len(blue) if key else 0

        start = float(blue[0]['timestamp_ms'])
        end = float(blue[-1]['timestamp_ms'])

        events.append({
            'event_id': len(events),
            'key': key,
            'start_ms': start,
            'end_ms': end,
            'observed_span_ms': end - start,
            'sampling_interval_ms': sampling_ms,
            'blue_frame_count': len(blue),
            'known_key_frames': sum(votes.values()),
            'vote_support': support,
            'status': (
                'resolved' if key
                else ('tie' if votes else 'unknown')
            )
        })

    return events

def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]
