import torch
import argparse

import os
import subprocess

def form_list_from_user_input(args):
    if args.file_with_video_paths is not None:
        with open(args.file_with_video_paths) as rfile:
            # remove carriage return
            path_list = [line.replace('\n', '') for line in rfile.readlines()]
            # remove empty lines
            path_list = [path for path in path_list if len(path) > 0]
    else:
        path_list = args.video_paths

    return path_list


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library
    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path

def parallel_feature_extraction(args):
    '''Distributes the feature extraction in a embarasingly-parallel fashion. Specifically,
    it divides the dataset (list of video paths) among all specified devices evenly.'''

    from i3d.extract_i3d import ExtractI3D  # defined here to avoid import errors
    extractor = ExtractI3D(args)

    video_paths = form_list_from_user_input(args)
    indices = torch.arange(len(video_paths))
    replicas = torch.nn.parallel.replicate(extractor, args.device_ids[:len(indices)])
    inputs = torch.nn.parallel.scatter(indices, args.device_ids[:len(indices)])
    torch.nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)
    extractor.progress.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Features')
    # Main args
    parser.add_argument('--feature_type', default='i3d')
    parser.add_argument('--video_paths', nargs='+', help='space-separated paths to videos')
    parser.add_argument('--file_with_video_paths', help='.txt file where each line is a path')
    parser.add_argument('--device_ids', type=int, nargs='+', help='space-separated device ids')
    parser.add_argument('--tmp_path', default='./tmp',
                        help='folder to store the extracted frames before the extraction')
    parser.add_argument('--keep_frames', dest='keep_frames', action='store_true', default=False,
                        help='to keep frames after feature extraction')
    parser.add_argument('--on_extraction', default='save_numpy',
                        help='what to do once the stack is extracted')
    parser.add_argument('--output_path', default='./output', help='where to store results if saved')
    # I3D options
    parser.add_argument('--pwc_path', default='./i3d/checkpoints/pwc_net.pt')
    parser.add_argument('--i3d_rgb_path', default='./i3d/checkpoints/i3d_rgb.pt')
    parser.add_argument('--i3d_flow_path', default='./i3d/checkpoints/i3d_flow.pt')
    parser.add_argument('--min_side_size', type=int, default=256, help='min(HEIGHT, WIDTH)')
    parser.add_argument('--extraction_fps', type=int, help='Do not specify for original video fps')
    parser.add_argument('--stack_size', type=int, default=64, help='Feature time span in fps')
    parser.add_argument('--step_size', type=int, default=64, help='Feature step size in fps')
    
    args = parser.parse_args()

    # some printing
    if args.on_extraction == 'save_numpy':
        print(f'Saving features to {args.output_path}')
    if args.keep_frames:
        print(f'Keeping temp files in {args.tmp_path}')

    parallel_feature_extraction(args)
