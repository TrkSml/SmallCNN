
// __author__ = Tarek Samaali

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
//#include <windows.h>
#include <ctype.h>
#include <stdint.h>
#include "utils.h"


void model_code(){

    //A model to hold the different layers
    Model* model;

    //A weight stack to save the weight change
    weight_stack* weightStack=NULL;


    // A random input as image
    Block* X;
    create_Block(&X,3,80,80,"random","float");

    // A target
    // the first parameter is the number of classes
    // the second parameter is the corresponding class
    Grid* Y=fill_index(12,11);

    // number of epochs
    uint16_t epochs=20;
    uint16_t counter;

    for(counter=0;counter<epochs;counter++){

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

    display_Grid(model->final_layer->output_data->grid);

    if(!counter)
    summary_layers(&model,"forward");

    backpropagation(&model);

    weightStack=model->weightStack;

    }

}


int main()
{

    //Debugging the code
    model_code();

    printf("\nDONE :))) ! \n\n");

    return 0;
}



