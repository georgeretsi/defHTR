
classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '


cdict = {c:i for i,c in enumerate(classes)}
icdict = {i:c for i,c in enumerate(classes)}

cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
cnn_top = 256

max_epochs = 240
batch_size = 20

level = "line"
fixed_size = (4 * 32, 4 * 256)

