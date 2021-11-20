import os
import pandas as pd

csv_path = 'Data/public_training_data.csv'
save_train = 'Data/train.txt'
save_valid = 'Data/valid.txt'
img_dir = 'Data/IMAGES/'

if not os.path.exists(img_dir):
	os.makedirs(img_dir)

csv_ = pd.read_csv(csv_path)
num_data = len(csv_['filename'])

split_num = 400

train_txt = []
valid_txt = []
for i,fn in enumerate(csv_['filename']):
	boxes = [csv_['top left x'][i],csv_['top left y'][i],csv_['bottom right x'][i],csv_['bottom right y'][i]]
	boxes = [str(round(b)) for b in boxes]
	boxes = ','.join(boxes)
	if i==500:
		break
	if i<split_num:	
		train_txt.append(os.path.join(img_dir,fn+'.jpg')+' '+boxes+ ',0\n')
	else:
		valid_txt.append(os.path.join(img_dir,fn+'.jpg')+' '+boxes+ ',0\n')

print(len(valid_txt),len(train_txt))
f = open(save_train,'w')
f.writelines(train_txt)
f.close()

f = open(save_valid,'w')
f.writelines(valid_txt)
f.close()

