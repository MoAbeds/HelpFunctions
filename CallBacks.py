
import tensorflow as tf
import datetime


def create_tensorboard_callback():
    
  experiment_name = input('Please Enter Your Experiment Name')
  log_dir = "tensorboard" + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback



def Check_Point():
    CheckPoint_name= input('Please Enter your CheckPoint Name')
    Check_Point_Path = f'model_checkPoints/{CheckPoint_name}.ckpt'
    model_checkPoints = tf.keras.callbacks.ModelCheckpoint(Check_Point_Path,montior = 'val_acc',save_best_only = True,save_weights_only=True,verbose=0)
    print(model_checkPoints)
    return model_checkPoints

def Early_Stop():
    early_Stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience =3)
    return early_Stop

def Reduce_lr():
  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2,verbose=1,min_lr=1e-7)
  return reduce_lr