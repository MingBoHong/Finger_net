import alexnet
import preprocess
import evaluate
import tensorflow as tf
import os
import numpy as np

max_iter_num = 100000  # 设置参数迭代次数
checkpoint_path = 'model/checkpoint'  # 设置模型参数文件所在路径
event_log_path = 'event-log'  # 设置事件文件所在路径，用于周期性存储Summary缓存对象
tfrecords = 'train.tfrecords'
batch_size = 3
def train():
    with tf.Graph().as_default():  # 指定当前图为默认graph
        global_step = tf.Variable(initial_value=0,trainable=False)

        img_batch, label_batch = preprocess.read_and_decode(tfrecords,batch_size,flag=True)
        Finger_model = alexnet.alexNet(img_batch,0.5,2,True)
        logits = Finger_model.fc_2
        total_loss = alexnet.loss(logits, label_batch)  # 计算损失
        one_step_gradient_update = alexnet.one_step_train(total_loss, global_step)  # 返回一步梯度更新操作
        # 创建一个saver对象，用于保存参数到文件中
        saver = tf.train.Saver(var_list=tf.all_variables())  # tf.all_variables return a list of `Variable` objects
        all_summary_obj = tf.summary.merge_all()  # 返回所有summary对象先merge再serialize后的的字符串类型tensor
        initiate_variables = tf.initialize_all_variables()

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(initiate_variables)  # 变量初始化
            tf.train.start_queue_runners(sess=sess)  # 启动所有的queuerunners
            if not os.path.exists(event_log_path):
                os.makedirs(event_log_path)
            Event_writer = tf.summary.FileWriter(logdir=event_log_path, graph=sess.graph)
            for step in range(max_iter_num):
                _, loss_value = sess.run(fetches=[one_step_gradient_update, total_loss])
                assert not np.isnan(loss_value)  # 用于验证当前迭代计算出的loss_value是否合理

                if step % 100 == 0:
                    # 添加`Summary`协议缓存到事件文件中，故不能写total_loss变量到事件文件中，因为这里的total_loss为普通的tensor类型
                    print('step %d, the loss_value is %.2f' % (step, loss_value))
                    all_summaries = sess.run(all_summary_obj)
                    Event_writer.add_summary(summary=all_summaries, global_step=step)
                if step % 1000 == 0 or (step + 1) == max_iter_num:
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    variables_save_path = os.path.join(checkpoint_path, 'model-parameters.bin')  # 路径合并，返回合并后的字符串
                    saver.save(sess, variables_save_path,
                               global_step=step)  # 把所有变量（包括moving average前后的模型参数）保存在variables_save_path路径下
                    evaluate.evaluate()


if __name__ == '__main__':
    train()