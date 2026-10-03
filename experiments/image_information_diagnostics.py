"""Model-free image information measurements using Pillow and the standard library."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, __version__ as PILLOW_VERSION

from eden_vision.services.image_input import ImageInputService


ANALYSIS_MAX_SIDE = 512
DARK_LEVEL = 10
BRIGHT_LEVEL = 245
EDGE_GRADIENT_LEVEL = 20.0
DEFINITIONS = {
    'kind': 'raw_image_information_diagnostic; not confidence or a rejection rule',
    'decoder': 'Existing ImageInputService: EXIF transpose, RGB8, alpha composited on RGB(128,128,128), MPO frame 0, HEIC/HEIF support',
    'analysis_resize': 'Preserve aspect ratio; Pillow LANCZOS thumbnail max side 512; never upscale. All metrics use this raster.',
    'dimensions': 'width/height are decoded oriented source dimensions; analysis_width/height are measurement raster dimensions',
    'luminance': 'Pillow RGB.convert(L): quantized 8-bit luma approximating 0.299R+0.587G+0.114B on encoded RGB, not linear-light luminance',
    'mean_luminance': 'Arithmetic mean of grayscale values, range 0..255',
    'std_luminance': 'Population standard deviation of grayscale values; grayscale_std is an exact alias',
    'dark_pixel_fraction': 'Fraction grayscale <= 10; near_black_fraction is an exact alias',
    'bright_pixel_fraction': 'Fraction grayscale >= 245; near_white_fraction is an exact alias',
    'percentile_01': '1st percentile, linear interpolation at sorted index 0.01*(N-1)',
    'percentile_99': '99th percentile, linear interpolation at sorted index 0.99*(N-1)',
    'dynamic_range': 'Maximum grayscale minus minimum grayscale, in 8-bit levels',
    'robust_dynamic_range': 'percentile_99 minus percentile_01',
    'entropy': 'Shannon entropy -sum(p*log2(p)) of 256-bin grayscale histogram, bits, range 0..8',
    'detail_metric': 'Population variance of signed 4-neighbor Laplacian L=left+right+up+down-4*center; exclude 1-pixel border; units grayscale-level squared',
    'edge_density': 'Fraction of interior pixels with sqrt(gx^2+gy^2)>=20, gx=(right-left)/2, gy=(down-up)/2; exclude 1-pixel border',
    'small_image_policy': 'For width or height <3 at analysis scale, detail_metric and edge_density are null: no interior pixels',
    'saturation_mean': 'Mean Pillow HSV S channel divided by 255, range 0..1',
    'saturation_std': 'Population std of Pillow HSV S channel divided by 255',
    'fixed_metric_constants': {'analysis_max_side': ANALYSIS_MAX_SIDE, 'dark_lte': DARK_LEVEL,
                               'bright_gte': BRIGHT_LEVEL, 'edge_gradient_gte': EDGE_GRADIENT_LEVEL},
    'unknown_thresholds': None,
    'versions': {'pillow': PILLOW_VERSION},
}


def decode_rgb(data: bytes) -> Image.Image:
    decoded = ImageInputService(max_upload_bytes=100 * 1024 * 1024,
                                max_image_pixels=50_000_000).decode(data, None)
    return Image.frombytes('RGB', (decoded.width, decoded.height), decoded.pixels)


def histogram_stats(histogram):
    count = sum(histogram)
    if count <= 0:
        raise ValueError('empty image histogram')
    mean = sum(level * count_at_level for level, count_at_level in enumerate(histogram)) / count
    variance = sum(count_at_level * (level - mean) ** 2 for level, count_at_level in enumerate(histogram)) / count
    return mean, math.sqrt(variance)


def histogram_percentile(histogram, percentile):
    count = sum(histogram)
    if count == 0 or not 0 <= percentile <= 1:
        raise ValueError('invalid histogram or percentile')
    index = (count - 1) * percentile
    lower, upper = math.floor(index), math.ceil(index)
    values = []
    for target in (lower, upper):
        cumulative = 0
        for level, amount in enumerate(histogram):
            cumulative += amount
            if cumulative > target:
                values.append(level)
                break
    return values[0] + (index - lower) * (values[1] - values[0])


def measure_rgb(image: Image.Image) -> dict:
    if image.width <= 0 or image.height <= 0:
        raise ValueError('image must have positive dimensions')
    if image.mode != 'RGB':
        raise ValueError('measure_rgb requires decoder-normalized RGB; use measure_bytes for other modes')
    analysis = image.copy()
    analysis.thumbnail((ANALYSIS_MAX_SIDE, ANALYSIS_MAX_SIDE), Image.Resampling.LANCZOS)
    gray = analysis.convert('L')
    histogram = gray.histogram()
    n = gray.width * gray.height
    mean, std = histogram_stats(histogram)
    p01, p99 = (histogram_percentile(histogram, p) for p in (.01, .99))
    nonempty = [level for level, amount in enumerate(histogram) if amount]
    entropy = -sum((count / n) * math.log2(count / n) for count in histogram if count)
    pixels = gray.tobytes()
    w, height = gray.size
    lap_sum = lap_sq_sum = edge_count = interior = 0
    for y in range(1, height - 1):
        for x in range(1, w - 1):
            i = y * w + x
            left, right, up, down = pixels[i - 1], pixels[i + 1], pixels[i - w], pixels[i + w]
            lap = left + right + up + down - 4 * pixels[i]
            lap_sum += lap
            lap_sq_sum += lap * lap
            # Central differences: division by 2 is moved to the constant.
            edge_count += (right - left) ** 2 + (down - up) ** 2 >= 4 * EDGE_GRADIENT_LEVEL ** 2
            interior += 1
    saturation_mean, saturation_std = histogram_stats(analysis.convert('HSV').getchannel('S').histogram())
    return {
        'width': image.width, 'height': image.height,
        'analysis_width': w, 'analysis_height': height,
        'mean_luminance': mean, 'std_luminance': std, 'grayscale_std': std,
        'dark_pixel_fraction': sum(histogram[:DARK_LEVEL + 1]) / n,
        'bright_pixel_fraction': sum(histogram[BRIGHT_LEVEL:]) / n,
        'near_black_fraction': sum(histogram[:DARK_LEVEL + 1]) / n,
        'near_white_fraction': sum(histogram[BRIGHT_LEVEL:]) / n,
        'percentile_01': p01, 'percentile_99': p99,
        'dynamic_range': nonempty[-1] - nonempty[0], 'robust_dynamic_range': p99 - p01,
        'entropy': entropy,
        'detail_metric': max(0., lap_sq_sum / interior - (lap_sum / interior) ** 2) if interior else None,
        'edge_density': edge_count / interior if interior else None,
        'detail_status': 'available' if interior else 'not_evaluable: no interior pixels',
        'saturation_mean': saturation_mean / 255, 'saturation_std': saturation_std / 255,
    }


def measure_bytes(data: bytes) -> dict:
    with decode_rgb(data) as image:
        return measure_rgb(image)


def measure_path(path: Path) -> dict:
    return measure_bytes(Path(path).read_bytes())
