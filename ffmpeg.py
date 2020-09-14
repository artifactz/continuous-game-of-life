import subprocess

class FfmpegVideoWriter:
    def __init__(self, filename, width, height, fps, input_pixfmt='rgba', crf=24, output_pixfmt='yuv420p', video_filter=None):
        args = [
            'ffmpeg',
            '-hwaccel', 'auto',
            '-f', 'rawvideo',
            '-pixel_format', input_pixfmt,
            '-video_size', '{}x{}'.format(width, height),
            '-framerate', '{}'.format(fps),
            '-i', '-',
        ]
        if video_filter is not None:
            args += [
                '-filter:v', video_filter,
            ]
        args += [
            '-vc', 'libx264',
            '-crf', str(crf),
            '-preset', 'veryslow',
            '-x264-params', 'opencl=true',
            '-pix_fmt', output_pixfmt,
            '-movflags', '+faststart',
            '-y',
            filename
        ]
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE)

    def encode_image(self, img):
        self.p.stdin.write(img)

    def close(self):
        self.p.terminate()
