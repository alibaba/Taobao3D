# make osx, ios, visionos all together
import argparse
import os
import shutil
import subprocess
import sys

root_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + '/../..').replace('\\', '/')
ios_toolchain_path = root_dir + '/scripts/cmake/ios.toolchain.cmake'

def log_info(log):
    print(f'[INFO] {log}')

def log_error(log):
    print(f'[ERROR] {log}')

def delete_build_dir(platform):
    build_dir = f'{root_dir}/build/{platform}'
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

def make_project(platform, cmd):
    # create build dir
    build_dir = f'{root_dir}/build/{platform}'
    os.makedirs(build_dir, exist_ok=True)

    cmd.insert(0, 'cmake')
    cmd.append(f'-S{root_dir}')
    cmd.append(f'-B{build_dir}')
    log_info('cmd: ' + ' '.join(cmd))

    ret = subprocess.call(cmd, cwd=build_dir)
    if (ret == 0):
        log_info('Succeed making project at path: {}'.format(build_dir))
    else:
        log_error('Fail to make project, check errors above.')
        sys.exit(1)

def make_project_osx():
    cmd = [
        '-G Xcode',
        '-DTARGET_PLATFORM=PLATFORM_OSX'
    ]
    make_project('osx', cmd)

def make_project_ios():
    cmd = [
        '-G Xcode',
        f'-DCMAKE_TOOLCHAIN_FILE={ios_toolchain_path}',
        '-DPLATFORM=OS64COMBINED',
        '-DDEPLOYMENT_TARGET=14.0',
        '-DTARGET_PLATFORM=PLATFORM_IOS'
    ]

    make_project('ios', cmd)

def make_project_visionos():
    cmd = [
        '-G Xcode',
        f'-DCMAKE_TOOLCHAIN_FILE={ios_toolchain_path}',
        '-DPLATFORM=VISIONOSCOMBINED',
        '-DDEPLOYMENT_TARGET=2.0',
        '-DTARGET_PLATFORM=PLATFORM_VISIONOS'
        # '-DCMAKE_CXX_FLAGS="-stdlib=libc++"'
    ]

    # generate project by cmake
    make_project('visionos', cmd)

def build_dependencies(platform):
    #mnn
    mnn_root = './third/mnn'
    if platform == 'osx':
        subprocess.call('./package_scripts/mac/buildFrameWork.sh', cwd = mnn_root)
    elif platform == 'ios':
        ## TODO::change to ios only
        subprocess.call('./package_scripts/ios/buildiOS_withsimulator.sh', cwd = mnn_root)
    elif platform == 'visionos':
        subprocess.call('../mnn_build_visionos.sh', cwd = mnn_root)

def check_dependencies(platform):
    if platform == 'osx':
        return os.path.exists(f'{root_dir}/third/mnn/MNN-MacOS-CPU-GPU')
    elif platform == 'ios':
        return os.path.exists(f'{root_dir}/third/mnn/MNN-iOS-CPU-GPU')
    elif platform == 'visionos':
        return os.path.exists(f'{root_dir}/third/mnn/MNN-visionOS-CPU-GPU') 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remake', action='store_true', help='remove build dir to remake project')
    parser.add_argument('--platform', type=str, default='osx', help='supported platforms: ios, osx, visionos')
    parser.add_argument('--rebuild_dependency', action='store_true', help='rebuild third party dependencies')
    args = parser.parse_args()

    if args.rebuild_dependency or not check_dependencies(args.platform):
        build_dependencies(args.platform)
        
    if args.remake:
        delete_build_dir(args.platform)

    if args.platform == 'osx':
        make_project_osx()
    elif args.platform == 'ios':
        make_project_ios()
    elif args.platform == 'visionos':
        make_project_visionos()
    else:
        log_error(f'unknown platform name: {args.platform}')
        sys.exit(1)


if __name__ == '__main__':
    main()