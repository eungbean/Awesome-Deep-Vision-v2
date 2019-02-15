
# Awesome Deep Vision [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Maintainers - [Eungbean Lee](https://eungbean.github.io)

A curated list of deep learning resources for computer vision, inspired by [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision) - Not Maintained after 2017

I am taking this list as a milestone for studying computer vision fields.
I would like to inform you that I have rewritten the latest papers and important papers.

## References
- [Awesome Deep Learning](https://github.com/terryum/awesome-deep-learning-papers) by [Terry Um](https://www.facebook.com/terryum.io/)
- [Awesome Deep Vision](https://github.com/kjw0612/awesome-deep-vision) by [Jiwon Kim](https://github.com/kjw0612), [Heesoo Myeong](https://github.com/hmyeong), [Myungsub Choi](https://github.com/myungsub), [Jung Kwon Lee](https://github.com/deruci), [Taeksoo Kim](https://github.com/jazzsaxmafia)
- [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision) by [Jia-Bin Huang](http://www.jiabinhuang.com)
- [Research Paper Reading List](https://github.com/txizzle/reading-list) by [Ted Xiao](http://www.tedxiao.me)



## Table of Contents
- [General Resources](#general-resources)
  - [Paper lists](#paper-lists)
  - [Courses](#courses)
  - [Lab Blogs](#lab-blogs)
  - [Personal Blogs](#personal-blogs)
  - [Books](#books)
  - [Videos](#videos)
  - [Papers](#papers)
    - [Image processing](#image-processing)
      - [Filtering](#filtering)
  - [ImageNet Classification](#imagenet-classification)
  - [Object Detection](#object-detection)
  - [Object Tracking](#object-tracking)
  - [Low-Level Vision](#low-level-vision)
    - [Super-Resolution](#super-resolution)
    - [Stereo Matching](#stereo-matching)
    - [Other Applications](#other-applications)
  - [Edge Detection](#edge-detection)
  - [Semantic Segmentation](#semantic-segmentation)
  - [Visual Attention and Saliency](#visual-attention-and-saliency)
  - [Object Recognition](#object-recognition)
  - [Human Pose Estimation](#human-pose-estimation)
  - [Understanding CNN](#understanding-cnn)
  - [Image and Language](#image-and-language)
    - [Image Captioning](#image-captioning)
    - [Video Captioning](#video-captioning)
    - [Question Answering](#question-answering)
  - [Image Generation](#image-generation)
  - [Other Topics](#other-topics)
- [Software](#software)
  - [Framework](#framework)
  - [Applications](#applications)
- [Tutorials](#tutorials)

<!--more-->
## General Resources

### Paper Lists

- [Awesome Deep Learning](https://github.com/terryum/awesome-deep-learning-papers)
- [Deep Learning Papers Reading Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap)
- [Awesome Deep Vision](https://github.com/kjw0612/awesome-deep-vision)
- [Deep Reinforcement Learning Papers](https://github.com/junhyukoh/deep-reinforcement-learning-papers)
- [ML@B Summer Reading List](https://docs.google.com/spreadsheets/d/1921snepdp5iQMqTfHic7fOtcgaXH27XK9MBS993cQXg/edit#gid=0)

### Courses
- Machine learning
  * [UC Berkeley] [CS 189/289A: Introduction to Machine Learning](http://www-inst.eecs.berkeley.edu/~cs189/fa15/), Fall 2015, [UC Berkeley login required]
  * [EE 221A: Nonlinear Systems](https://inst.eecs.berkeley.edu/~ee222/sp17/), UC Berkeley, Fall 2016
- Deep Vision
  * [Stanford] [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
  * [CUHK] [ELEG 5040: Advanced Topics in Signal Processing(Introduction to Deep Learning)](https://piazza.com/cuhk.edu.hk/spring2015/eleg5040/home)
- More Deep Learning
  * [Stanford] [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)
  * [Oxford] [Deep Learning by Prof. Nando de Freitas](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)
  * [NYU] [Deep Learning by Prof. Yann LeCun](http://cilvr.cs.nyu.edu/doku.php?id=courses:deeplearning2014:start)
  * [IEOR 265: Learning and Optimization](http://ieor.berkeley.edu/~aaswani/teaching/SP16/265/), UC Berkeley, Spring 2016
  * [CS 294-129: Designing, Visualizing, and Understanding Deep Neural Networks](https://bcourses.berkeley.edu/courses/1453965), UC Berkeley, Fall 2016 [UC Berkeley login required]
  * [CS 294-112: Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcoursesp17/), UC Berkeley,  Spring 2017
  * [CS 294-131: Special Topics in Deep Learning](https://berkeley-deep-learning.github.io/cs294-131-s17/), UC Berkeley, Spring 2017
  * [Deep Learning for everybody (Korean)](https://hunkim.github.io/ml/), by [Hun Kim](https://github.com/hunkim)

### Lab Blogs

- [BAIR blog](http://bair.berkeley.edu/blog/)
- [DeepMind blog](https://deepmind.com/blog/)
- [OpenAI blog](https://blog.openai.com/)
- [Google Research blog](https://research.googleblog.com/)
- [gBrain blog](https://research.googleblog.com/search/label/Google%20Brain)
- [FAIR blog](https://research.fb.com/blog/)
- [NVIDIA blog](https://blogs.nvidia.com/blog/category/deep-learning/)
- [MSR blog](https://www.microsoft.com/en-us/research/blog/)
- [Facebook's AI Painting@Wired](http://www.wired.com/2015/06/facebook-googles-fake-brains-spawn-new-visual-reality/)

### Personal Blogs

- [Lilian Weng, OpenAI](https://lilianweng.github.io/lil-log/)
- [Eric Jang, Robotics at Google](http://evjang.com/articles.html)
- [Alex Irpan, Robotics at Google](https://www.alexirpan.com/)

#### Old but useful posts
  - [Deep down the rabbit hole: CVPR 2015 and beyond@Tombone's Computer Vision Blog](http://www.computervisionblog.com/2015/06/deep-down-rabbit-hole-cvpr-2015-and.html)
  - [CVPR recap and where we're going@Zoya Bylinskii (MIT PhD Student)'s Blog](http://zoyathinks.blogspot.kr/2015/06/cvpr-recap-and-where-were-going.html)
  - [Inceptionism: Going Deeper into Neural Networks@Google Research](http://googleresearch.blogspot.kr/2015/06/inceptionism-going-deeper-into-neural.html)
  - [Implementing Neural networks](http://peterroelants.github.io/)

  ### Books
  -  Free Online Books
    * [Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](http://www.iro.umontreal.ca/~bengioy/dlbook/)
    * [Neural Networks and Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
    * [Deep Learning Tutorial by LISA lab, University of Montreal](http://deeplearning.net/tutorial/deeplearning.pdf)
  - Books
    * [Grokking Deep Learning for Computer Vision](https://www.manning.com/books/grokking-deep-learning-for-computer-vision)

### Videos
- Talks
  * [Deep Learning, Self-Taught Learning and Unsupervised Feature Learning By Andrew Ng](https://www.youtube.com/watch?v=n1ViNeWhC24)
  * [Recent Developments in Deep Learning By Geoff Hinton](https://www.youtube.com/watch?v=vShMxxqtDDs)
  * [The Unreasonable Effectiveness of Deep Learning by Yann LeCun](https://www.youtube.com/watch?v=sc-KbuZqGkI)
  * [Deep Learning of Representations by Yoshua bengio](https://www.youtube.com/watch?v=4xsVFLnHC_0)
  * [Siraj Raval](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A)
  * [Terry's Deep Learning Talk (Korean)][https://www.youtube.com/playlist?list=PL0oFI08O71gKEXITQ7OG2SCCXkrtid7Fq] by [Terry](https://www.facebook.com/deeplearningtalk)
  - Video Series
    * [Deep Learning Crash Course by Leo Isikdogan](https://www.youtube.com/watch?v=nmnaO6esC7c&list=PLWKotBjTDoLj3rXBL-nEIPRN9V3a9Cx07)
    * [Hands-on Deep Learning: TensorFlow Coding Sessions](https://www.youtube.com/watch?v=1KzJbIFnVTE&list=PLWKotBjTDoLhcczRktdYukFDU3BwXRNaN)


## Papers
### Image processing
* [-] SIFT (IJCV 2004), DG Lowe [[Paper]](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
* [-] HOG (CVPR 2005) N Dalal [[Paper]](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)

#### Filtering
* [v] Bilateral Filter (1998), C Tomasi [[Paper]](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/10spring/lectures/handouts/lec14_bilateral_4up.pdf)
* [v] Guided Filter (ECCV 2010), Kaiming He [[Paper]](http://kaiminghe.com/publications/eccv10guidedfilter.pdf) [[Project]](http://kaiminghe.com/eccv10/)
* [v] Rolling Guidance Filter (ECCV 2014), Q Zhang [[Paper]](Rolling Guidance Filter
https://pdfs.semanticscholar.org/.../c4000f5c71c22fb4a22fcf5dd0...) [[Project]](http://www.cse.cuhk.edu.hk/leojia/projects/rollguidance/)
* [v] WLS Filter (SIGGRAPH 2008), Z Farbman [[Paper]](http://evasion.imag.fr/Enseignement/cours/2009/ProjetImage/multiscale/multiscale.pdf) [[Projects]](http://www.cs.huji.ac.il/~danix/epd/)

### ImageNet Classification
![classification](https://cloud.githubusercontent.com/assets/5226447/8451949/327b9566-2022-11e5-8b34-53b4a64c13ad.PNG)
(from Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, ImageNet Classification with Deep Convolutional Neural Networks, NIPS, 2012.)

* [-] Deep Residual Learning (arXiv:1512.03385), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun [[Paper](http://arxiv.org/pdf/1512.03385v1.pdf)][[Slide](http://image-net.org/challenges/talks/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)]
* [-] PReLu/Weight Initialization (arXiv:1502.01852), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun [[Paper]](http://arxiv.org/pdf/1502.01852)
* [-] Batch Normalization (arXiv:1502.03167), Sergey Ioffe, Christian Szegedy [[Paper]](http://arxiv.org/pdf/1502.03167)
* [-] GoogLeNet (CVPR 2015), Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich [[Paper]](http://arxiv.org/pdf/1409.4842)
* [-] VGG-Net (ICLR 2015), Karen Simonyan and Andrew Zisserman [[Web]](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) [[Paper]](http://arxiv.org/pdf/1409.1556)
* [-] AlexNet (NIPS, 2012) Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, [[Paper]](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-25-2012)

### Object Detection
![object_detection](https://cloud.githubusercontent.com/assets/5226447/8452063/f76ba500-2022-11e5-8db1-2cd5d490e3b3.PNG)
(from Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, arXiv:1506.01497.)

* [-] PVANET (arXiv:1608.08021), Kye-Hyeon Kim [[Paper]](https://arxiv.org/pdf/1608.08021) [[Code]](https://github.com/sanghoon/pva-faster-rcnn)
* [-] OverFeat (ICLR 2014), P Sermanet [[Paper]](http://arxiv.org/pdf/1312.6229.pdf)
* [-] R-CNN, (CVPR 2014), Ross Girshick [[Paper-CVPR14]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) [[Paper-arXiv14]](http://arxiv.org/pdf/1311.2524)
* [-] SPP: Spatial Pyramid Pooling (ECCV 2014), Kaiming He [[Paper]](http://arxiv.org/pdf/1406.4729)
* [-] Fast R-CNN, (arXiv:1504.08083), Ross Girshick [[Paper]](http://arxiv.org/pdf/1504.08083)
* [-] Faster R-CNN, (arXiv:1506.01497), Shaoqing Ren [[Paper]](http://arxiv.org/pdf/1506.01497)
* [-] R-CNN minus R, (arXiv:1506.06981), Karel Lenc [[Paper]](http://arxiv.org/pdf/1506.06981)
* [-] End-to-end people detection in crowded scenes (arXiv:1506.04878), Russell Stewart, [[Paper]](http://arxiv.org/abs/1506.04878)
* [-] YOLO: Real-Time Object Detection, Joseph Redmon [[Project]](https://pjreddie.com/yolo/) [[C Code]](https://github.com/pjreddie/darknet), [[TF Code]](https://github.com/thtrieu/darkflow)
  * [-] You Only Look Once: Unified, Real-Time Object Detection (arXiv:1506.02640), Joseph Redmon [[Paper]](https://arxiv.org/abs/1506.02640)
  * [-] YOLO v2 (arXiv:1612.08242), Joseph Redmon [[Paper]](https://arxiv.org/abs/1612.08242)
  * [-] YOLOv3: An Incremental Improvement (1804.02767), Joseph Redmon [[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

* [-] Inside-Outside Net (arXiv:1512.04143), Sean Bell [[Paper](http://arxiv.org/abs/1512.04143)]
* [-] Deep Residual Network (arXiv:1512.03385), Kaiming He [[Paper](http://arxiv.org/abs/1512.03385)]
* [-] Weakly Supervised Object Localization with Multi-fold Multiple Instance Learning (arXiv:1503.00949), [[Paper](http://arxiv.org/pdf/1503.00949.pdf)]
* [-] R-FCN (arXiv:1605.06409), Jifeng Dai [[Paper](https://arxiv.org/abs/1605.06409)] [[Code](https://github.com/daijifeng001/R-FCN)]
* [-] SSD: Single Shot MultiBox Detector (arXiv:1512.05325v2) [[Paper]](https://arxiv.org/pdf/1512.02325v2.pdf) [[Code](https://github.com/weiliu89/caffe/tree/ssd)]
* [-] Speed/accuracy trade-offs for modern convolutional object detectors (arXiv:1611.10012), Jonarhan NHuang [[Paper](https://arxiv.org/pdf/1611.10012v1.pdf)]

### Video Classification
* [-] Delving Deeper into Convolutional Networks for Learning Video Representations (ICLR 2016), Nicolas Ballas [[Paper](http://arxiv.org/pdf/1511.06432v4.pdf)]
* [-] Deep Multi Scale Video Prediction Beyond Mean Square Error (ICLR 2016), Michael Mathieu [[Paper](http://arxiv.org/pdf/1511.05440v6.pdf)]

### Object Tracking
* [-] Online Tracking by Learning Discriminative Saliency Map with Convolutional Neural Network (arXiv:1502.06796), Seunghoon Hong [[Paper]](http://arxiv.org/pdf/1502.06796)
* [-] DeepTrack: Learning Discriminative Feature Representations by Convolutional Neural Networks for Visual Tracking, (BMVC  2014), Hanxi Li [[Paper]](http://www.bmva.org/bmvc/2014/files/paper028.pdf)
* [-] Learning a Deep Compact Image Representation for Visual Tracking, (NIPS 2013), N Wang [[Paper]](http://winsty.net/papers/dlt.pdf)
* [-] Hierarchical Convolutional Features for Visual Tracking (ICCV 2015), Chao Ma [[Paper](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Ma_Hierarchical_Convolutional_Features_ICCV_2015_paper.pdf)] [[Code](https://github.com/jbhuang0604/CF2)]
* [-] Visual Tracking with fully Convolutional Networks (ICCV 2015), Lijun Wang [[Paper](http://202.118.75.4/lu/Paper/ICCV2015/iccv15_lijun.pdf)] [[Code](https://github.com/scott89/FCNT)]
* [-] Learning Multi-Domain Convolutional Neural Networks for Visual Tracking (arXiv:1510.07945), Hyeonseob Nam, Bohyung Han [[Paper](http://arxiv.org/pdf/1510.07945.pdf)] [[Code](https://github.com/HyeonseobNam/MDNet)] [[Project Page](http://cvlab.postech.ac.kr/research/mdnet/)]

### Low-Level Vision

#### Super-Resolution
* [-] Iterative Image Reconstruction (IJCAI, 2001), Sven Behnke [[Paper]](http://www.ais.uni-bonn.de/behnke/papers/ijcai01.pdf)
* [-] SRCNN: Super-Resolution (ECCV 2014), Chao Dong [[Web]](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) [[Paper-ECCV14]](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf) [[Paper-arXiv15]](http://arxiv.org/pdf/1501.00092.pdf)
* [-] Very Deep Super-Resolution (arXiv:1511.04587), Jiwon Kim [[Paper]](http://arxiv.org/abs/1511.04587)
* [-] Deeply-Recursive Convolutional Network (arXIv:1511.04491), Jiwon Kim [[Paper]](http://arxiv.org/abs/1511.04491)
* [-] Casade-Sparse-Coding-Network (ICCV 2015), Zhaowen Wang [[Paper]](http://www.ifp.illinois.edu/~dingliu2/iccv15/iccv15.pdf) [[Code]](http://www.ifp.illinois.edu/~dingliu2/iccv15/)
* [-] Perceptual Losses for Super-Resolution (arXiv:1603.08155), Justin Johnson [[Paper]](http://arxiv.org/abs/1603.08155) [[Supplementary]](http://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf)
* [-] SRGAN (arXiv:1609.04802v3), Christian Ledig [[Paper]](https://arxiv.org/pdf/1609.04802v3.pdf)
* [-] Image Super-Resolution with Fast Approximate Convolutional Sparse Coding (ICONIP 2014), Osendorfer [[Paper 2014]](http://brml.org/uploads/tx_sibibtex/281.pdf)

#### Stereo Matching
* [-] Convolutional Neural Network to Compare Image Patches (arXiv:1510.05970), S Zagoruyko [[Paper]](https://arxiv.org/abs/1510.05970)
* [-] FlowNet (arXiv:1504.06852), P Fischer [[Paper]](https://arxiv.org/abs/1504.06852),
* [-] Discriminative Learning of Deep Convolutional Feature Point Descriptors (ICCV 2015), E Simo-Serra [[Paper]](https://icwww.epfl.ch/~trulls/pdf/iccv-2015-deepdesc.pdf), 2015,
* [-] MatchNet (CVPR 2015), X Han [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Han_MatchNet_Unifying_Feature_2015_CVPR_paper.pdf),
* [-] Computing the Stereo Matching Cost with a Convolutional Neural Network (CVPR 2015), Jure Žbontar [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zbontar_Computing_the_Stereo_2015_CVPR_paper.pdf)
* [-] Colorful Image Colorization (ECCV 2016), Richard Zhang [[Paper]](http://arxiv.org/pdf/1603.08511.pdf), [[Code]](https://github.com/richzhang/colorization)


#### Other Applications
* [-] Optical Flow (FlowNet) (arXiv:1504.06852) , Philipp Fischer [[Paper]](http://arxiv.org/pdf/1504.06852)
* [-] Compression Artifacts Reduction (arXiv:1504.06993), Chao Dong [[Paper]](http://arxiv.org/pdf/1504.06993)

- Blur Removal
  * [-] Learning to Deblur, (arXiv:1406.7444), Christian J. Schuler [[Paper]](http://arxiv.org/pdf/1406.7444.pdf)
  * [-] Learning a Convolutional Neural Network for Non-uniform Motion Blur Removal, (CVPR 2015), Jian Sun [[Paper]](http://arxiv.org/pdf/1503.00593)
* [-] Image Deconvolution (NIPS 2014), Li Xu [[Web]](http://lxu.me/projects/dcnn/) [[Paper]](http://lxu.me/mypapers/dcnn_nips14.pdf)
* [-] Deep Edge-Aware Filter (ICMR 2015), Li Xu [[Paper]](http://jmlr.org/proceedings/papers/v37/xub15.pdf)
* [-] Colorful Image Colorization (ECCV 2016), [[Paper]](https://arxiv.org/abs/1603.08511) [[Project]](http://richzhang.github.io/colorization/) [[Blog-Ryan Dahl]](http://tinyclouds.org/colorize/)
* [-] Feature Learning by Inpainting (CVPR 2016), Deepak Pathak [[Paper]](https://arxiv.org/pdf/1604.07379v1.pdf)[[Code]](https://github.com/pathak22/context-encoder)

### Edge Detection
![edge_detection](https://cloud.githubusercontent.com/assets/5226447/8452371/93ca6f7e-2025-11e5-90f2-d428fd5ff7ac.PNG)
(from Gedas Bertasius, Jianbo Shi, Lorenzo Torresani, DeepEdge: A Multi-Scale Bifurcated Deep Network for Top-Down Contour Detection, CVPR, 2015.)

* [-] Holistically-Nested Edge Detection (arXiv:1504.06375), Saining Xie [[Paper]](http://arxiv.org/pdf/1504.06375) [[Code]](https://github.com/s9xie/hed)
* [-] DeepEdge (CVPR 2015), Gedas Bertasius [[Paper]](http://arxiv.org/pdf/1412.1123)
* [-] DeepContour (CVPR 2015), Wei Shen [[Paper]](http://mc.eistar.net/UpLoadFiles/Papers/DeepContour_cvpr15.pdf)

### Semantic Segmentation
![semantic_segmantation](https://cloud.githubusercontent.com/assets/5226447/8452076/0ba8340c-2023-11e5-88bc-bebf4509b6bb.PNG)
(from Jifeng Dai, Kaiming He, Jian Sun, BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation, arXiv:1503.01640.)
* [-] PASCAL VOC2012 Challenge Leaderboard (01 Sep. 2016)
  ![VOC2012_top_rankings](https://cloud.githubusercontent.com/assets/3803777/18164608/c3678488-7038-11e6-9ec1-74a1542dce13.png)
  (from PASCAL VOC2012 [leaderboards](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6))
* [-] SEC: Seed, Expand and Constrain (ECCV 2016), Alexander Kolesnikov [[Paper]](http://pub.ist.ac.at/~akolesnikov/files/ECCV2016/main.pdf) [[Code]](https://github.com/kolesman/SEC)
* [-] Adelaide, Guosheng Lin
  * [-] Efficient piecewise training of deep structured models for semantic segmentation (arXiv:1504.01013), Guosheng Lin  [[Paper]](http://arxiv.org/pdf/1504.01013) (1st ranked in VOC2012)
  * [-] Deeply Learning the Messages in Message Passing Inference (arXiv:1508.02108), Guosheng Lin [[Paper]](http://arxiv.org/pdf/1506.02108) (4th ranked in VOC2012)

* [-] Deep Parsing Network (DPN) (ICCV 2015), Ziwei Liu [[Paper]](http://arxiv.org/pdf/1509.02634.pdf) (2nd ranked in VOC 2012)
* [-] CentraleSuperBoundaries (arXiv 1511.07386), Iasonas Kokkinos [[Paper]](http://arxiv.org/pdf/1511.07386) (4th ranked in VOC 2012)
* [-] BoxSup (arXiv:1503.01640), ifeng Dai, [[Paper]](http://arxiv.org/pdf/1503.01640) (6th ranked in VOC2012)
- POSTECH
  * [-] Learning Deconvolution Network for Semantic Segmentation (arXiv:1505.04366), Hyeonwoo Noh [[Paper]](http://arxiv.org/pdf/1505.04366) (7th ranked in VOC2012)
  * [-] Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation (arXiv:1506.04924), Seunghoon Hong [[Paper]](http://arxiv.org/pdf/1506.04924)
  * [-] Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network (arXiv:1512.07928), Seunghoon Hong [[Paper](http://arxiv.org/pdf/1512.07928.pdf)] [[Project Page](http://cvlab.postech.ac.kr/research/transfernet/)]
* [-] Conditional Random Fields as Recurrent Neural Networks (arXiv:1502.03240), Shuai Zheng [[Paper]](http://arxiv.org/pdf/1502.03240) (8th ranked in VOC2012)
* [-] DeepLab (arXiv:1502.02734), Liang-Chieh Chen [[Paper]](http://arxiv.org/pdf/1502.02734) (9th ranked in VOC2012)
* [-] Zoom-out (CVPR 2015), Mohammadreza Mostajabi [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mostajabi_Feedforward_Semantic_Segmentation_2015_CVPR_paper.pdf)
* [-] Joint Calibration(arXiv:1507.01581), Holger Caesar [[Paper]](http://arxiv.org/pdf/1507.01581)
* [-] Fully Convolutional Networks for Semantic Segmentation (CVPR 2015), Jonathan Long [[Paper-CVPR15]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) [[Paper-arXiv15]](http://arxiv.org/pdf/1411.4038)
* [-] Hypercolumn, Bharath Hariharan (CVPR 2015) [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hariharan_Hypercolumns_for_Object_2015_CVPR_paper.pdf)
* [-] Deep Hierarchical Parsing (CVPR 2015), Abhishek Sharma [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sharma_Deep_Hierarchical_Parsing_2015_CVPR_paper.pdf)
* [-] Learning Hierarchical Features for Scene Labeling, Clement Farabet [[Paper-ICML12]](http://yann.lecun.com/exdb/publis/pdf/farabet-icml-12.pdf) [[Paper-PAMI13]](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)
- SegNet [[Web]](http://mi.eng.cam.ac.uk/projects/segnet/)
  * [-] SegNet (arXiv:1511.00561), Vijay Badrinarayanan [[Paper]](http://arxiv.org/abs/1511.00561)
  * [-] Bayesian SegNet (arXiv:1511.02680), Alex Kendall [[Paper]](http://arxiv.org/abs/1511.00561)
* [-] Multi-Scale Context Aggregation by Dilated Convolutions (ICLR 2016), Fisher Yu [[Paper](http://arxiv.org/pdf/1511.07122v2.pdf)]
* [-] Segment-Phrase Table for Semantic Segmentation, Visual Entailment and Paraphrasing, (ICCV 2015), Hamid Izadinia [[Paper](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Izadinia_Segment-Phrase_Table_for_ICCV_2015_paper.pdf)]
* [-] Pusing the Boundaries of Boundary Detection Using deep Learning (ICLR 2016), Iasonas Kokkinos [[Paper](http://arxiv.org/pdf/1511.07386v2.pdf)]
* [-] Weakly supervised graph based semantic segmentation by learning communities of image-parts (ICCV 2015), Niloufar Pourian [[Paper](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Pourian_Weakly_Supervised_Graph_ICCV_2015_paper.pdf)]

### Visual Attention and Saliency
![saliency](https://cloud.githubusercontent.com/assets/5226447/8492362/7ec65b88-2183-11e5-978f-017e45ddba32.png)
(from Nian Liu, Junwei Han, Dingwen Zhang, Shifeng Wen, Tianming Liu, Predicting Eye Fixations using Convolutional Neural Networks, CVPR, 2015.)

* [-] Mr-CNN (CVPR 2015), Nian Liu [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Predicting_Eye_Fixations_2015_CVPR_paper.pdf)
* [-] Learning a Sequential Search for Landmarks (CVPR 2015), Saurabh Singh [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Singh_Learning_a_Sequential_2015_CVPR_paper.pdf)
* [-] Multiple Object Recognition with Visual Attention (ICLR 2015), Jimmy Lei Ba [[Paper]](http://arxiv.org/pdf/1412.7755.pdf)
* [-] Recurrent Models of Visual Attention (NIPS 2014), Volodymyr Mnih [[Paper]](http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)

### Object Recognition
* [-] Weakly-supervised learning with convolutional neural networks (CVPR 2015), Maxime Oquab [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Oquab_Is_Object_Localization_2015_CVPR_paper.pdf)
* [-] FV-CNN (CVPR 2015), Mircea Cimpoi [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Cimpoi_Deep_Filter_Banks_2015_CVPR_paper.pdf)

### Human Pose Estimation
* [-] Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (CVPR 2017), Zhe Cao
* [-] Deepcut: Joint subset partition and labeling for multi person pose estimation (CVPR 2016), Leonid Pishchulin
* [-] Convolutional pose machines (CVPR 2016), Shih-En Wei
* [-] Stacked hourglass networks for human pose estimation (ECCV 2016), Alejandro Newell
* [-] Flowing convnets for human pose estimation in videos (ICCV 2015), T Pfister
* [-] Joint training of a convolutional network and a graphical model for human pose estimation (NIPS 2014), Jonathan J. Tompson

### Understanding CNN
![understanding](https://cloud.githubusercontent.com/assets/5226447/8452083/1aaa0066-2023-11e5-800b-2248ead51584.PNG)
(from Aravindh Mahendran, Andrea Vedaldi, Understanding Deep Image Representations by Inverting Them, CVPR, 2015.)

* [-] Understanding image representations by measuring their equivariance and equivalence (CVPR 2015), Karel Lenc [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Lenc_Understanding_Image_Representations_2015_CVPR_paper.pdf)
* [-] Deep Neural Networks are Easily Fooled:High Confidence Predictions for Unrecognizable Images (CVPR 2015), Anh Nguyen [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Nguyen_Deep_Neural_Networks_2015_CVPR_paper.pdf)
* [-] Understanding Deep Image Representations by Inverting Them (CVPR 2015), Aravindh Mahendran [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mahendran_Understanding_Deep_Image_2015_CVPR_paper.pdf)
* [-] Object Detectors Emerge in Deep Scene CNNs (ICLR 2015), Bolei Zhou [[arXiv Paper]](http://arxiv.org/abs/1412.6856)
* [-] Inverting Visual Representations with Convolutional Networks (arXiv:1506.02753), Alexey Dosovitskiy [[Paper]](http://arxiv.org/abs/1506.02753)
* [-] Visualizing and Understanding Convolutional Networks, (ECCV 2014), Matthrew Zeiler [[Paper]](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)

### Image and Language

#### Image Captioning
![image_captioning](https://cloud.githubusercontent.com/assets/5226447/8452051/e8f81030-2022-11e5-85db-c68e7d8251ce.PNG)
(from Andrej Karpathy, Li Fei-Fei, Deep Visual-Semantic Alignments for Generating Image Description, CVPR, 2015.)

* UCLA / Baidu [[Paper]](http://arxiv.org/pdf/1410.1090)
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Alan L. Yuille, Explain Images with Multimodal Recurrent Neural Networks, arXiv:1410.1090.
* Toronto [[Paper]](http://arxiv.org/pdf/1411.2539)
  * Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel, Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models, arXiv:1411.2539.
* Berkeley [[Paper]](http://arxiv.org/pdf/1411.4389)
  * Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, Trevor Darrell, Long-term Recurrent Convolutional Networks for Visual Recognition and Description, arXiv:1411.4389.
* Google [[Paper]](http://arxiv.org/pdf/1411.4555)
  * Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan, Show and Tell: A Neural Image Caption Generator, arXiv:1411.4555.
* Stanford [[Web]](http://cs.stanford.edu/people/karpathy/deepimagesent/) [[Paper]](http://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
  * Andrej Karpathy, Li Fei-Fei, Deep Visual-Semantic Alignments for Generating Image Description, CVPR, 2015.
* UML / UT [[Paper]](http://arxiv.org/pdf/1412.4729)
  * Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, Kate Saenko, Translating Videos to Natural Language Using Deep Recurrent Neural Networks, NAACL-HLT, 2015.
* CMU / Microsoft [[Paper-arXiv]](http://arxiv.org/pdf/1411.5654) [[Paper-CVPR]](http://www.cs.cmu.edu/~xinleic/papers/cvpr15_rnn.pdf)
  * Xinlei Chen, C. Lawrence Zitnick, Learning a Recurrent Visual Representation for Image Caption Generation, arXiv:1411.5654.
  * Xinlei Chen, C. Lawrence Zitnick, Mind’s Eye: A Recurrent Visual Representation for Image Caption Generation, CVPR 2015
* Microsoft [[Paper]](http://arxiv.org/pdf/1411.4952)
  * Hao Fang, Saurabh Gupta, Forrest Iandola, Rupesh Srivastava, Li Deng, Piotr Dollár, Jianfeng Gao, Xiaodong He, Margaret Mitchell, John C. Platt, C. Lawrence Zitnick, Geoffrey Zweig, From Captions to Visual Concepts and Back, CVPR, 2015.
* Univ. Montreal / Univ. Toronto [[Web](http://kelvinxu.github.io/projects/capgen.html)] [[Paper](http://www.cs.toronto.edu/~zemel/documents/captionAttn.pdf)]
  * Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard S. Zemel, Yoshua Bengio, Show, Attend, and Tell: Neural Image Caption Generation with Visual Attention, arXiv:1502.03044 / ICML 2015
* Idiap / EPFL / Facebook [[Paper](http://arxiv.org/pdf/1502.03671)]
  * Remi Lebret, Pedro O. Pinheiro, Ronan Collobert, Phrase-based Image Captioning, arXiv:1502.03671 / ICML 2015
* UCLA / Baidu [[Paper](http://arxiv.org/pdf/1504.06692)]
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, Alan L. Yuille, Learning like a Child: Fast Novel Visual Concept Learning from Sentence Descriptions of Images, arXiv:1504.06692
* MS + Berkeley
  * Jacob Devlin, Saurabh Gupta, Ross Girshick, Margaret Mitchell, C. Lawrence Zitnick, Exploring Nearest Neighbor Approaches for Image Captioning, arXiv:1505.04467 [[Paper](http://arxiv.org/pdf/1505.04467.pdf)]
  * Jacob Devlin, Hao Cheng, Hao Fang, Saurabh Gupta, Li Deng, Xiaodong He, Geoffrey Zweig, Margaret Mitchell, Language Models for Image Captioning: The Quirks and What Works, arXiv:1505.01809 [[Paper](http://arxiv.org/pdf/1505.01809.pdf)]
* Adelaide [[Paper](http://arxiv.org/pdf/1506.01144.pdf)]
  * Qi Wu, Chunhua Shen, Anton van den Hengel, Lingqiao Liu, Anthony Dick, Image Captioning with an Intermediate Attributes Layer, arXiv:1506.01144
* Tilburg [[Paper](http://arxiv.org/pdf/1506.03694.pdf)]
  * Grzegorz Chrupala, Akos Kadar, Afra Alishahi, Learning language through pictures, arXiv:1506.03694
* Univ. Montreal [[Paper](http://arxiv.org/pdf/1507.01053.pdf)]
  * Kyunghyun Cho, Aaron Courville, Yoshua Bengio, Describing Multimedia Content using Attention-based Encoder-Decoder Networks, arXiv:1507.01053
* Cornell [[Paper](http://arxiv.org/pdf/1508.02091.pdf)]
  * Jack Hessel, Nicolas Savva, Michael J. Wilber, Image Representations and New Domains in Neural Image Captioning, arXiv:1508.02091
* MS + City Univ. of HongKong [[Paper](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yao_Learning_Query_and_ICCV_2015_paper.pdf)]
  * Ting Yao, Tao Mei, and Chong-Wah Ngo, "Learning Query and Image Similarities
    with Ranking Canonical Correlation Analysis", ICCV, 2015

#### Video Captioning
* Berkeley [[Web]](http://jeffdonahue.com/lrcn/) [[Paper]](http://arxiv.org/pdf/1411.4389.pdf)
  * Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, Trevor Darrell, Long-term Recurrent Convolutional Networks for Visual Recognition and Description, CVPR, 2015.
* UT / UML / Berkeley [[Paper]](http://arxiv.org/pdf/1412.4729)
  * Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, Kate Saenko, Translating Videos to Natural Language Using Deep Recurrent Neural Networks, arXiv:1412.4729.
* Microsoft [[Paper]](http://arxiv.org/pdf/1505.01861)
  * Yingwei Pan, Tao Mei, Ting Yao, Houqiang Li, Yong Rui, Joint Modeling Embedding and Translation to Bridge Video and Language, arXiv:1505.01861.
* UT / Berkeley / UML [[Paper]](http://arxiv.org/pdf/1505.00487)
  * Subhashini Venugopalan, Marcus Rohrbach, Jeff Donahue, Raymond Mooney, Trevor Darrell, Kate Saenko, Sequence to Sequence--Video to Text, arXiv:1505.00487.
* Univ. Montreal / Univ. Sherbrooke [[Paper](http://arxiv.org/pdf/1502.08029.pdf)]
  * Li Yao, Atousa Torabi, Kyunghyun Cho, Nicolas Ballas, Christopher Pal, Hugo Larochelle, Aaron Courville, Describing Videos by Exploiting Temporal Structure, arXiv:1502.08029
* MPI / Berkeley [[Paper](http://arxiv.org/pdf/1506.01698.pdf)]
  * Anna Rohrbach, Marcus Rohrbach, Bernt Schiele, The Long-Short Story of Movie Description, arXiv:1506.01698
* Univ. Toronto / MIT [[Paper](http://arxiv.org/pdf/1506.06724.pdf)]
  * Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, Sanja Fidler, Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books, arXiv:1506.06724
* Univ. Montreal [[Paper](http://arxiv.org/pdf/1507.01053.pdf)]
  * Kyunghyun Cho, Aaron Courville, Yoshua Bengio, Describing Multimedia Content using Attention-based Encoder-Decoder Networks, arXiv:1507.01053
* TAU / USC [[paper](https://arxiv.org/pdf/1612.06950.pdf)]
  * Dotan Kaufman, Gil Levi, Tal Hassner, Lior Wolf, Temporal Tessellation for Video Annotation and Summarization, arXiv:1612.06950.

#### Question Answering
![question_answering](https://cloud.githubusercontent.com/assets/5226447/8452068/ffe7b1f6-2022-11e5-87ab-4f6d4696c220.PNG)
(from Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, Devi Parikh, VQA: Visual Question Answering, CVPR, 2015 SUNw:Scene Understanding workshop)

* Virginia Tech / MSR [[Web]](http://www.visualqa.org/) [[Paper]](http://arxiv.org/pdf/1505.00468)
  * Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, Devi Parikh, VQA: Visual Question Answering, CVPR, 2015 SUNw:Scene Understanding workshop.
* MPI / Berkeley [[Web]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/visual-turing-challenge/) [[Paper]](http://arxiv.org/pdf/1505.01121)
  * Mateusz Malinowski, Marcus Rohrbach, Mario Fritz, Ask Your Neurons: A Neural-based Approach to Answering Questions about Images, arXiv:1505.01121.
* Toronto [[Paper]](http://arxiv.org/pdf/1505.02074) [[Dataset]](http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/)
  * Mengye Ren, Ryan Kiros, Richard Zemel, Image Question Answering: A Visual Semantic Embedding Model and a New Dataset, arXiv:1505.02074 / ICML 2015 deep learning workshop.
* Baidu / UCLA [[Paper]](http://arxiv.org/pdf/1505.05612) [[Dataset]]()
  * Hauyuan Gao, Junhua Mao, Jie Zhou, Zhiheng Huang, Lei Wang, Wei Xu, Are You Talking to a Machine? Dataset and Methods for Multilingual Image Question Answering, arXiv:1505.05612.
* POSTECH [[Paper](http://arxiv.org/pdf/1511.05756.pdf)] [[Project Page](http://cvlab.postech.ac.kr/research/dppnet/)]
  * Hyeonwoo Noh, Paul Hongsuck Seo, and Bohyung Han, Image Question Answering using Convolutional Neural Network with Dynamic Parameter Prediction, arXiv:1511.05765
* CMU / Microsoft Research [[Paper](http://arxiv.org/pdf/1511.02274v2.pdf)]
  * Yang, Z., He, X., Gao, J., Deng, L., & Smola, A. (2015). Stacked Attention Networks for Image Question Answering. arXiv:1511.02274.
* MetaMind [[Paper](http://arxiv.org/pdf/1603.01417v1.pdf)]
  * Xiong, Caiming, Stephen Merity, and Richard Socher. "Dynamic Memory Networks for Visual and Textual Question Answering." arXiv:1603.01417 (2016).
* SNU + NAVER [[Paper](http://arxiv.org/abs/1606.01455)]
  * Jin-Hwa Kim, Sang-Woo Lee, Dong-Hyun Kwak, Min-Oh Heo, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang, *Multimodal Residual Learning for Visual QA*, arXiv:1606:01455
* UC Berkeley + Sony [[Paper](https://arxiv.org/pdf/1606.01847)]
  * Akira Fukui, Dong Huk Park, Daylen Yang, Anna Rohrbach, Trevor Darrell, and Marcus Rohrbach, *Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding*, arXiv:1606.01847
* Postech [[Paper](http://arxiv.org/pdf/1606.03647.pdf)]
  * Hyeonwoo Noh and Bohyung Han, *Training Recurrent Answering Units with Joint Loss Minimization for VQA*, arXiv:1606.03647
* SNU + NAVER [[Paper](http://arxiv.org/abs/1610.04325)]
  * Jin-Hwa Kim, Kyoung Woon On, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang, *Hadamard Product for Low-rank Bilinear Pooling*, arXiv:1610.04325.

  ### Image Generation
  - Convolutional / Recurrent Networks
    * [-] Conditional Image Generation with PixelCNN Decoders (arXiv:1606.05328v2), Aäron van den Oord [[Paper]](https://arxiv.org/pdf/1606.05328v2.pdf)[[Code]](https://github.com/kundan2510/pixelCNN)
    * [-]  Learning to Generate Chairs with Convolutional Neural Networks (CVPR 2015), Alexey Dosovitskiy [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf)
    * [-]  DRAW: A Recurrent Neural Network For Image Generation (ICML 2015), Karol Gregor [[Paper](https://arxiv.org/pdf/1502.04623v2.pdf)]
  - Adversarial Networks
    * [-] GAN: Generative Adversarial Network (NIPS 2014), Ian J. Goodfellow. [[Paper]](http://arxiv.org/abs/1406.2661)
    * [-] Deep Generative Image Models using a Laplacian Pyramid of Adversarial Network (NIPS 2015), Emily Denton [[Paper]](http://arxiv.org/abs/1506.05751)
    * [-] A note on the evaluation of generative models (ICLR 2016), Lucas Theis [[Paper](http://arxiv.org/abs/1511.01844)]
    * [-] Variationally Auto-Encoded Deep Gaussian Processes (ICLR 2016), Zhenwen Dai [[Paper](http://arxiv.org/pdf/1511.06455v2.pdf)]
    * [-] Generating Images from Captions with Attention (ICLR 2016), Elman Mansimov [[Paper](http://arxiv.org/pdf/1511.02793v2.pdf)]
    * [-] Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks (ICLR 2016), Jost Tobias Springenberg [[Paper](http://arxiv.org/pdf/1511.06390v1.pdf)]
    * [-] Censoring Representations with an Adversary (ICLR 2016), Harrison Edwards [[Paper](http://arxiv.org/pdf/1511.05897v3.pdf)]
    * [-] Distributional Smoothing with Virtual Adversarial Training (ICLR 2016), Takeru Miyato [[Paper](http://arxiv.org/pdf/1507.00677v8.pdf)]
    * [-] Generative Visual Manipulation on the Natural Image Manifold (ECCV 2016), Jun-Yan Zhu [[Paper](https://arxiv.org/pdf/1609.03552v2.pdf)] [[Code](https://github.com/junyanz/iGAN)] [[Video](https://youtu.be/9c4z6YsBGQ0)]
    * [-] Mixing Convolutional and Adversarial Networks (ICLR 2016), Alec Radford [[Paper](http://arxiv.org/pdf/1511.06434.pdf)]

    ### Other Topics
    * Visual Analogy [[Paper](https://web.eecs.umich.edu/~honglak/nips2015-analogy.pdf)]
      * Scott Reed, Yi Zhang, Yuting Zhang, Honglak Lee, Deep Visual Analogy Making, NIPS, 2015
    * Surface Normal Estimation [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Designing_Deep_Networks_2015_CVPR_paper.pdf)
      * Xiaolong Wang, David F. Fouhey, Abhinav Gupta, Designing Deep Networks for Surface Normal Estimation, CVPR, 2015.
    * Action Detection [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Gkioxari_Finding_Action_Tubes_2015_CVPR_paper.pdf)
      * Georgia Gkioxari, Jitendra Malik, Finding Action Tubes, CVPR, 2015.
    * Crowd Counting [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Cross-Scene_Crowd_Counting_2015_CVPR_paper.pdf)
      * Cong Zhang, Hongsheng Li, Xiaogang Wang, Xiaokang Yang, Cross-scene Crowd Counting via Deep Convolutional Neural Networks, CVPR, 2015.
    * 3D Shape Retrieval [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Sketch-Based_3D_Shape_2015_CVPR_paper.pdf)
      * Fang Wang, Le Kang, Yi Li, Sketch-based 3D Shape Retrieval using Convolutional Neural Networks, CVPR, 2015.
    * Weakly-supervised Classification
      * Samaneh Azadi, Jiashi Feng, Stefanie Jegelka, Trevor Darrell, "Auxiliary Image Regularization for Deep CNNs with Noisy Labels", ICLR 2016, [[Paper](http://arxiv.org/pdf/1511.07069v2.pdf)]
    * Artistic Style [[Paper]](http://arxiv.org/abs/1508.06576) [[Code]](https://github.com/jcjohnson/neural-style)
      * Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, A Neural Algorithm of Artistic Style.
    * Human Gaze Estimation
      * Xucong Zhang, Yusuke Sugano, Mario Fritz, Andreas Bulling, Appearance-Based Gaze Estimation in the Wild, CVPR, 2015. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Appearance-Based_Gaze_Estimation_2015_CVPR_paper.pdf) [[Website]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild-mpiigaze/)
    * Face Recognition
      * Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf, DeepFace: Closing the Gap to Human-Level Performance in Face Verification, CVPR, 2014. [[Paper]](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)
      * Yi Sun, Ding Liang, Xiaogang Wang, Xiaoou Tang, DeepID3: Face Recognition with Very Deep Neural Networks, 2015. [[Paper]](http://arxiv.org/abs/1502.00873)
      * Florian Schroff, Dmitry Kalenichenko, James Philbin, FaceNet: A Unified Embedding for Face Recognition and Clustering, CVPR, 2015. [[Paper]](http://arxiv.org/abs/1503.03832)
    * Facial Landmark Detection
      * Yue Wu, Tal Hassner, KangGeon Kim, Gerard Medioni, Prem Natarajan, Facial Landmark Detection with Tweaked Convolutional Neural Networks, 2015. [[Paper]](http://arxiv.org/abs/1511.04031) [[Project]](http://www.openu.ac.il/home/hassner/projects/tcnn_landmarks/)


        ## Software
        ### Framework
        * Tensorflow: An open source software library for numerical computation using data flow graph by Google [[Web](https://www.tensorflow.org/)]
        * Torch7: Deep learning library in Lua, used by Facebook and Google Deepmind [[Web](http://torch.ch/)]
          * Torch-based deep learning libraries: [[torchnet](https://github.com/torchnet/torchnet)],
        * Caffe: Deep learning framework by the BVLC [[Web](http://caffe.berkeleyvision.org/)]
        * Theano: Mathematical library in Python, maintained by LISA lab [[Web](http://deeplearning.net/software/theano/)]
          * Theano-based deep learning libraries: [[Pylearn2](http://deeplearning.net/software/pylearn2/)], [[Blocks](https://github.com/mila-udem/blocks)], [[Keras](http://keras.io/)], [[Lasagne](https://github.com/Lasagne/Lasagne)]
        * MatConvNet: CNNs for MATLAB [[Web](http://www.vlfeat.org/matconvnet/)]
        * MXNet: A flexible and efficient deep learning library for heterogeneous distributed systems with multi-language support [[Web](http://mxnet.io/)]
        * Deepgaze: A computer vision library for human-computer interaction based on CNNs [[Web](https://github.com/mpatacchiola/deepgaze)]

        ### Applications
        * Adversarial Training
          * Code and hyperparameters for the paper "Generative Adversarial Networks" [[Web]](https://github.com/goodfeli/adversarial)
        * Understanding and Visualizing
          * Source code for "Understanding Deep Image Representations by Inverting Them," CVPR, 2015. [[Web]](https://github.com/aravindhm/deep-goggle)
        * Semantic Segmentation
          * Source code for the paper "Rich feature hierarchies for accurate object detection and semantic segmentation," CVPR, 2014. [[Web]](https://github.com/rbgirshick/rcnn)
          * Source code for the paper "Fully Convolutional Networks for Semantic Segmentation," CVPR, 2015. [[Web]](https://github.com/longjon/caffe/tree/future)
        * Super-Resolution
          * Image Super-Resolution for Anime-Style-Art [[Web]](https://github.com/nagadomi/waifu2x)
        * Edge Detection
          * Source code for the paper "DeepContour: A Deep Convolutional Feature Learned by Positive-Sharing Loss for Contour Detection," CVPR, 2015. [[Web]](https://github.com/shenwei1231/DeepContour)
          * Source code for the paper "Holistically-Nested Edge Detection", ICCV 2015. [[Web]](https://github.com/s9xie/hed)

  ## Tutorials
  * [CVPR 2014] [Tutorial on Deep Learning in Computer Vision](https://sites.google.com/site/deeplearningcvpr2014/)
  * [CVPR 2015] [Applied Deep Learning for Computer Vision with Torch](https://github.com/soumith/cvpr2015)
