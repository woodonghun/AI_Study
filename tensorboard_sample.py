import math

from torch.utils.tensorboard import SummaryWriter

'''
    Terminal 에서 먼저 tensorboard --logdir=./main_folder
    logdir 는 아래의 log_dir 에 맞춰서 입력
    logdir 의 하위폴더에 있는 event 파일 즉 저장한 그래프 파일을 전부 출력한다.
    
    main.py 위치에서 폴더가 생성 된 것이 아니면 하위 폴더 까지 입력해야한다
        
    파일명은 변경 불가, 파일명 뒤에 주석처럼 붙이는 것은 가능.
'''

main_folder = 'main_folder'     # terminal 에서 입력해야 하는 최상위 폴더 명
project_folder_name = 'help'   # tensorboard 에서 분류 가능 할수 있도록 폴더를 만듬
tensorboard_file_name = ''  # 이벤트 파일의 뒤에 따로 붙일수 있다

# tensorboard
writer = SummaryWriter(log_dir=f'{main_folder}/{project_folder_name}', filename_suffix=tensorboard_file_name)

for step in range(-360, 360):
    angle_rad = step * math.pi / 180

    # 각각 그래프에 입력 scalar
    writer.add_scalar('sin', math.sin(angle_rad), step)
    writer.add_scalar('cos', math.cos(angle_rad), step)

    # 하나의 그래프에 2개의 선 입력
    # 하위 폴더가 생성됨
    writer.add_scalars('graph', {'sin': math.sin(angle_rad), 'cos': math.cos(angle_rad)}, step)

# 더 이상 추가할 scalar 값이 없으면 반드시 입력.
writer.flush()
writer.close()
