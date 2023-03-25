import streamlit as st 
import os 
import sys 
import numpy as np
from PIL import Image as PImage
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.general import get_data_root

import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import torch
from torchvision import transforms
st.set_option('deprecation.showfileUploaderEncoding', False)

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from torch.utils.model_zoo import load_url
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   

# Upload an image and set some options for demo purposes
st.header("Cropper Demo")

dataset = 'oxford5k'
cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
#Load vecs



img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

if img_file is None:
    st.write("Give me an image you want now and i will give you my heart <3 ")

if img_file:
    img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    
    # Manipulate cropped image at will
    
    st.write("Preview")
    cropped_img.save("data\demo\demo.jpg")
    _ = cropped_img.thumbnail((150,150))
    
    st.image(cropped_img)
    # st.write(cropped_img)

#operator = st.selectbox("Select cropped image",['YES','NO'])
# click = st.button("RUN")
if img_file is not None:    
    PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet50-gem-w'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
    }
    st.write('CHOOSE THE MODEL')
    choose = st.selectbox("Select the method:" ,("VGG16","ResNet101"))
    # choose = st.checkbox("Choose method",('ChooseVGG16','ChooseResNet101'))
    if choose =='VGG16':
        network_path = 'retrievalSfM120k-vgg16-gem'
        vecs = np.load('corpus/VGG/vector_VGGOx.npy')
    if choose =='ResNet101':
        network_path = 'retrievalSfM120k-resnet101-gem'
        vecs = np.load('corpus/ResNet/vector_ResOx.npy')
    if network_path in PRETRAINED:
        # pretrained networks (downloaded automatically)
        state = load_url(PRETRAINED[network_path], model_dir=os.path.join(get_data_root(), 'networks'))
    else:
        # fine-tuned network from path
        state = torch.load(network_path)
    n = st.number_input("Amount of result image returned",1,20,1)
    
    click = st.button("RUN")
    if click:
        # parsing net params from meta
        # architecture, pooling, mean, std required
        # the rest has default values, in case that is doesnt exist
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get('local_whitening', False)
        net_params['regional'] = state['meta'].get('regional', False)
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False

        # load network
        net = init_network(net_params)
        net.load_state_dict(state['state_dict'])

        # if whitening is precomputed
        if 'Lw' in state['meta']:
            net.meta['Lw'] = state['meta']['Lw']

        print(">>>> loaded network: ")
        print(net.meta_repr())

        # set up the transform
        normalize = transforms.Normalize(
            mean=net.meta['mean'],
            std=net.meta['std']
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        try:
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        except:
            bbxs = None  # for holidaysmanrot and copydays

        ms = [1, 1/2**(1/2), 1/2]
        if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
            msp = net.pool.p.item()
            print(">> Set-up multiscale:")
            print(">>>> ms: {}".format(ms))            
            print(">>>> msp: {}".format(msp))
        else:
            msp = 1
        dimages = ["data/demo/demo.jpg"]
        dvecs = extract_vectors(net, dimages, 1024, transform, bbxs=bbxs, ms=ms, msp=msp)
        scores = np.dot(vecs.T, dvecs)
        ranks = np.argsort(-scores, axis=0)

        for i in range(n):
            img = Image.open(images[ranks[i][0]])
            st.title("Rank: {rank}".format(rank=i + 1))
            
            st.image(img)

                