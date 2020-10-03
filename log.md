

1.对冰场数据111.txt进行处理，选取了大概前100行，然后将其导入excel中处理成类似项目原始数据中的形式
</br>
发现了一个bug
</br>
ValueError: invalid literal for int() with base 10: '393\t2563'

2. self.frame_pointer += self.seq_length
<br>
这一行将会导致数组越界，但是原本项目不存在这个问题

3.grid_seq = grids[dataloader.dataset_pointer][(num_batch * dataloader.batch_size) + sequence]
 <br>
 将train.py中的一个修改了不会报bug
 
4. 在可视化中将frame_str注释掉了

5.在使用grid的时候最后一个网格tensor加入限制


