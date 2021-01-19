### NUS Statistics 2021 Hackathon

Contains the code for yWaste's hackathon submission

#### Installation

The following project was built using python, and its associated machine learning and computer vision packages.
We used tensorflow for model training and python-opencv2 for image processing.
We used Anaconda for python virtual environment management.

After cloning the repo locally, run the following command to install all dependancies in virtual environment:  
`pip install -r requirements.txt`

Clone [labelImg](https://github.com/tzutalin/labelImg), a python based image annotions tool. Follow the instructions to install, and proceed to draw bounding boxes on the chips.

#### Open Issues

Since we are using gpu acceleration using CUDA, downloading on windows platform may have some issues for CUDA Toolkit and cuDNN libraries.

Please see the following issues regarding different config issues encountered:

[#44291](https://github.com/tensorflow/tensorflow/issues/44291) Enabling GPU support for tensorflow in windows (open, jan-21)

[#25138](https://github.com/tensorflow/tensorflow/issues/25138) Limit GPU memory on tensorflow (closed, jan-21)

#### References

- Sample [detection](https://stackoverflow.com/questions/39689938/python-opencv2-counting-contours-of-colored-object)

- [darknet13][1], Joseph Redmon, Darknet: Open source Neural Networks in C, 2013 - 2016

[1]: http://pjreddie.com/darknet/
