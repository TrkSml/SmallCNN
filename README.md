# SmallCNN

A minimalist engine for Deep and Shallow convolutional neural networks tried on a single random image / signe target. It allows to build layer-based convolutional models.

A snippet of a single training loop :



    create_Model(&model, // the model
                 &weightStack,  // the weight stack
                 X,  // the input
                 Y,  // the target
                 .2,  // the learning rate
                 20,  // number of epochs
                 counter); // the actual epoch
                 
    //A first convolution layer:
    //1st parameter is the depth of the input
    //2nd parameter is the stride
    //3rd parameter is the padding
    //4rth parameter is the kernel size;
    //5th parameter is the activation function ;

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
    

