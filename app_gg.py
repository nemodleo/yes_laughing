
# import gangealing, os
# os.chdir('gangealing')
# os.environ['PYTHONPATH'] = '/env/python:/content/gangealing'
# !pip install ninja ray plotly==4.14.3 torch==1.10.1 torchvision==0.11.2 imageio==2.4.1 --upgrade

# from applications.mixed_reality import run_gangealing_on_video
from applications import load_stn



model = 'celeba'
video = 'elon'
video_size = 512
video_path = 'assets/elon/'
object_picker = 'celeba_pokemon.png'


fps = 30 #@param {type:"integer"}
blend_alg = 'alpha'  #@param ["alpha", "laplacian", "laplacian_light"]
batch_size =  1#@param {type:"integer"}
use_flipping = False #@param {type:"boolean"}
memory_efficient_but_slower = False #@param {type:"boolean"}



class MyDict(): 
  def __init__(self): pass

# Assign a bunch of arguments. For some reason, this demo
# runs way faster when invoking python commands directly 
# than calling python from bash.
args = MyDict()
args.real_size = int(video_size)
args.real_data_path = video_path
args.fps = fps
args.batch = batch_size
args.blend_alg = blend_alg
args.transform = ['similarity', 'flow']
args.flow_size = 128
args.stn_channel_multiplier = 0.5
args.num_heads = 1
args.distributed = False  # Colab only uses 1 GPU
args.clustering = False
args.cluster = None
args.objects = True
args.no_flip_inference = not use_flipping
args.save_frames = memory_efficient_but_slower
args.overlay_congealed = False
args.ckpt = model
args.override = False
args.save_correspondences = False
args.out = 'visuals'
# # if object_picker.value == 'dense tracking':
# #   args.label_path = f'assets/masks/{model}_mask.png'
#     # Feel free to change the parameters below:
#     args.resolution = 128
#     args.sigma = 1.3
#     args.opacity = 0.8
#     args.objects = False
# else:  # object lense
args.label_path = f'assets/objects/{model}/{object_picker}' #.value}'
args.resolution = 4 * int(video_size)
args.sigma = 0.3
args.opacity = 1.0
args.objects = True

# Run Mixed Reality!
stn = load_stn(args)#, device='cpu')
#print(f'[*] {stn}', end='\n\n')

print('Running Spatial Transformer on frames...')



# run_gangealing_on_video(args, stn, classifier=None)

# print('Preparing videos to be displayed...')
# Display the output video: (snippet from https://stackoverflow.com/a/65273831)

# def postprocessing():
#     from glob import glob
#     import os
#     # from IPython.display import HTML
#     # from base64 import b64encode
#     num = len(list(glob(f'{video}_compressed*')))
#     compressed_name = f'{video}_compressed{num}.mp4'
#     congealed_compressed_name = f'{video}_compressed_congealed{num}.mp4'
#     path = f'visuals/video_{video}/propagated.mp4'
#     congealed_path = f'visuals/video_{video}/congealed.mp4'
#     os.system(f"ffmpeg -i {path} -vcodec libx264 {compressed_name}")
#     os.system(f"ffmpeg -i {congealed_path} -vcodec libx264 {congealed_compressed_name}")
#     mp4 = open(compressed_name,'rb').read()
#     mp4_congealed = open(congealed_compressed_name, 'rb').read()
#     # data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
#     # congealed_data_url = "data:video/mp4;base64," + b64encode(mp4_congealed).decode()
#     print(f'HIGHEST QUALITY output videos can be found at /content/gangealing/visuals/{video}')
#     # HTML("""<video width=512 autoplay controls loop><source src="%s" type="video/mp4"></video> <video width=512 autoplay controls loop><source src="%s" type="video/mp4"></video>""" % (data_url, congealed_data_url))

# postprocessing()
