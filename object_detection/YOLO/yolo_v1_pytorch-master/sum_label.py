import os

"""
    yolo 형식의 txt labeling data.txt 를 하나의 txt 로 묶는 코드
    
    코드 사용이유 : 강의에서 제공한 코드에서는 모든 labeling data 를 하나의 txt 로 묶어서 사용하기 때문
                  이 코드를 사용 한 뒤 yolo 에서 pascal 형식으로 변경되는데 그건 voc.py 에 정의되어 있다
                  
"""

def make_txt(dir):
    all_data = os.listdir(dir)
    txt_data = []
    jpg_data = []
    for i in all_data:
        if 'txt' in i:
            txt_data.append(i)
        else:
            jpg_data.append(i)

    for q in range(len(txt_data)):
        with open(dir+'/'+txt_data[q],"r") as f:
            string = f.readlines()
            for i in range(len(string)):
                string[i] = string[i].replace('\n',' ')

            with open(dir+'/voc2007.txt',"a") as t:
                t.write(jpg_data[q]+' ')
                t.writelines(string)
                t.write('\n')


if __name__ == '__main__':
    make_txt(r'C:\woo_project\AI_Study\object_detection\sample\train')