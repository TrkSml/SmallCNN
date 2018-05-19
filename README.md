# SmallCNN

A minimalist engine for Deep and Shallow convolutional neural networks tried on a single random image / signe target. It allows to build layer-based convolutional models.


`
create_Model(&model,
                 &weightStack,
                 X,
                 Y,
                 .2,
                 20,
                 counter);



    //A first convolution layer:
    //1st parameter is the depth of the input
    //2nd par° is the stride
    //3rd par° is the padding
    //4rth par° is the kernel size;
    //5th par° is the activation function ;

    add_CONV(&model,3,1,0,5,&relu);

    // The pooling layer takes respectively as input:
    // the padding, the stride, the kernel size and the pooling type
    add_POOL(&model,2,2,3,"max");
    add_CONV(&model,5,1,0,7,&relu);
    add_POOL(&model,1,1,3,"max");
    add_CONV(&model,10,1,0,5,&relu);
    add_POOL(&model,1,1,3,"max");
    add_CONV(&model,5,1,0,3,&relu);
    add_POOL(&model,1,1,3,"max");

    add_FLAT(&model);
    add_FCAF(&model,&tanh,100);
    add_FC(&model,&sigmoid,80);
    add_FC(&model,&sigmoid,60);
    add_FC(&model,&tanh,50);
    add_FC(&model,&tanh,20);
    add_FC(&model,&sigmoid,12);
    DENSE(&model);
    
`
