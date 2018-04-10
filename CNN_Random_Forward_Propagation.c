// __author__ = Tarek Samaali

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define UPPER_BOUND 10
#define GEN_RANDOM_SEED srand(time(NULL))
#define DEBUG prunisgned int f("debug !")
#define ERROR_DIMENSION_CONV prunisgned int f("Dimension conflict: Please review the dimensions of the convolved arrays! \n")
#define ERROR_DEPTH prunisgned int f("Cannot perform convolution: Please make sure the kernel and the block have the same depth. \n")
#define ERROR_NULL prunisgned int f("Cannot perform convolution: Please make sure both the kernel and the block are not null. \n")
#define ERROR_CREATING_BLOCKS prunisgned int f("Cannot create blocks : please make sure the length is > 1.\n");
#define ERROR_EVEN_DIMENSIONS prunisgned int f(" How about using a kernel with odd sizes :))) ! .. ");

#define current_Layer(x) prunisgned int f("\nCurrent Layer: %s\n",x)
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))
#define min(X, Y)  ((X) < (Y) ? (X) : (Y))

//2D output
//After single convolution
typedef struct {
    unisgned int  width;
    unisgned int  height;
    float** grid;

} Grid;

//3D Block for a single
typedef struct {
    unisgned int  depth;
    unisgned int  width;
    unisgned int  height;
    float*** matrix;
} Block;

//4D output
//After convolution with N filters
typedef struct{
    unisgned int length;
    Block** blocks;
}Blocks;

typedef struct{

    Grid* weights;
    Grid* Before_Activation;
    Grid* After_Activation;
    float (*activation)(float);

}FullyConnected;

float generate_random(){
    return ((float)rand())/((float)RAND_MAX) * UPPER_BOUND;

}

unisgned int  control_parity_kernel_size(unisgned int  size_kernel){
    return size_kernel%2==1;
}


unisgned int  determine_size_output(unisgned int  input_height, unisgned int  kernel_height, unisgned int  padding, unisgned int  stride){

     return (unisgned int )(((input_height-kernel_height+2*padding)/stride))+1;
}

float relu(float x){
    return max(0,x);
}

float sigmoid(float x){
    return 1./(1.+exp(-x));

}


void shape_block(Block* block){

    prunisgned int f("depth : %d \n",block->depth);
    prunisgned int f("height : %d \n",block->height);
    prunisgned int f("width : %d \n",block->width);

}

unisgned int  test_Block(Block* block){
    return block->height && block->width && block->depth;
}
void create_Grid(Grid** grid, unisgned int  input_height, unisgned int  input_width,char* choice){

    *grid=(Grid*)malloc(sizeof(Grid));
    (*grid)->height=input_height;
    (*grid)->width=input_width;


    if(choice=="random"){

        (*grid)->grid=(float**)malloc(input_height*sizeof(float*));
        unisgned int  counter_height;

        for(counter_height=0;counter_height<input_height;counter_height++){
            unisgned int  counter_width;
            float* row=(float*)malloc(input_width*(sizeof(float)));

            for(counter_width=0;counter_width<input_width;counter_width++){
                        *(row+counter_width)=(float)generate_random();
                }
            *((*grid)->grid+counter_height)=row;
        }


    }
    else if(choice=="zeros"){

       (*grid)->grid=(float**)malloc(input_height*sizeof(float*));
        unisgned int  counter_height;

        for(counter_height=0;counter_height<input_height;counter_height++){
            unisgned int  counter_width;
            float* row=(float*)malloc(input_width*(sizeof(float)));

            for(counter_width=0;counter_width<input_width;counter_width++){
                        *(row+counter_width)=0.0;
                }
            *((*grid)->grid+counter_height)=row;
        }


    }

}

void create_Block(Block** block, unisgned int  input_depth, unisgned int  input_height, unisgned int  input_width,char* choice){

    *block=(Block*)malloc(sizeof(Block));
    (*block)->height=input_height;
    (*block)->width=input_width;
    (*block)->depth=input_depth;

    (*block)->matrix=(float***)malloc(input_depth*sizeof(float**));

    unisgned int  index_depth;
    for(index_depth=0;index_depth<input_depth;index_depth++){

        Grid* grid;
        create_Grid(&grid,input_height,input_width,choice);
        *((*block)->matrix+index_depth)=grid->grid;

    }

}


void create_Blocks(Blocks **blocks, unisgned int  length, unisgned int  depth, unisgned int  height, unisgned int  width, char* choice){

    if(length<1){

        ERROR_CREATING_BLOCKS;
        exit(0);

    }else{

       *blocks=(Blocks *)malloc(sizeof(Blocks));
       Blocks* blocks_tmp=*blocks;
       blocks_tmp->blocks=malloc(length*sizeof(Block));
       unisgned int  index_length;

       for(index_length=0;index_length<length;index_length++){
            Block *new_block=(Block*)malloc(sizeof(Block));
            create_Block(&new_block,depth,height,width,choice);
            //display_Block(new_block);
            *(blocks_tmp->blocks+index_length)=new_block;
       }

       blocks_tmp->length=length;

    }
}




void apply_function_to_Grid(Grid** grid, float (*pounisgned int er_to_function)(float)){
    unisgned int  index_width,index_height;

    for(index_height=0;index_height<(*grid)->height;index_height++){
        for(index_width=0;index_width<(*grid)->width;index_width++){
                float* current_element=&((*grid)->grid[index_height][index_width]);
                ((*grid)->grid[index_height][index_width])=(*pounisgned int er_to_function)(*current_element);


        }
    }

}

//Extract a smaller Grid From a Grid
Grid* Extract_From_Grid(Grid* grid,\
                          unisgned int  begin_input_height, unisgned int  end_input_height,\
                          unisgned int  begin_input_width, unisgned int  end_input_width){

    Grid* output_grid=(Grid*)malloc(sizeof(Grid));

    unisgned int  size_height=end_input_height-begin_input_height;
    unisgned int  size_width=end_input_width-begin_input_width;

    output_grid->width=size_width;
    output_grid->height=size_height;

    output_grid->grid=(float**)malloc(output_grid->height*sizeof(float*));

    unisgned int  counter_height;
    unisgned int  new_counter_height=0;

    for(counter_height=begin_input_height;counter_height<end_input_height;counter_height++){

        unisgned int  counter_width;
        unisgned int  new_counter_width=0;
        float* row=(float*)malloc(output_grid->width*(sizeof(float)));

        for(counter_width=begin_input_width;counter_width<end_input_width;counter_width++){
                    *(row+new_counter_width)=grid->grid[counter_height][counter_width];
                    new_counter_width++;

            }

        output_grid->grid[new_counter_height]=row;
        new_counter_height++;
    }

    return output_grid;

}


Block* Extract_From_Block(Block* grid,\
                          unisgned int  begin_input_depth, unisgned int  end_input_depth,\
                          unisgned int  begin_input_height, unisgned int  end_input_height,\
                          unisgned int  begin_input_width, unisgned int  end_input_width){


    Block* output_grid=(Block*)malloc(sizeof(Block));

    unisgned int  size_depth=end_input_depth-begin_input_depth;
    unisgned int  size_height=end_input_height-begin_input_height;
    unisgned int  size_width=end_input_width-begin_input_width;

    output_grid->depth=size_depth;
    output_grid->width=size_width;
    output_grid->height=size_height;

    output_grid->matrix=(float***)malloc(size_depth*sizeof(float**));

    unisgned int  counter_depth;
    unisgned int  new_counter_depth=0;

    for(counter_depth=begin_input_depth;counter_depth<end_input_depth;counter_depth++){

                    Grid* grid_from_current_block=(Grid*)malloc(sizeof(Grid));
                    grid_from_current_block->height=size_height;
                    grid_from_current_block->width=size_width;
                    grid_from_current_block->grid=grid->matrix[counter_depth];

                    Grid* extracted_grid=Extract_From_Grid(grid_from_current_block,begin_input_height,end_input_height,\
                          begin_input_width,end_input_width);
                        *((output_grid->matrix)+new_counter_depth)=extracted_grid->grid;
                        new_counter_depth++;

                    free(grid_from_current_block);

                }

    return output_grid;

}


Grid* AddPadding_Grid(Grid** block,unisgned int  padding){

    Grid* output_grid;

    unisgned int  height=(*block)->height;
    unisgned int  width=(*block)->width;
    float **block_matrix=(*block)->grid;

    create_Grid(&output_grid,height+2*padding,width+2*padding,"zeros");

    //float** output_grid_matrix=output_grid->grid;

    unisgned int  counter_height;
    unisgned int  counter_width;


        for(counter_height=padding;counter_height<padding+height;counter_height++){

            for(counter_width=padding;counter_width<padding+width;counter_width++){

                output_grid->grid[counter_height][counter_width]=\
                    block_matrix[counter_height-padding][counter_width-padding];

            }
    }

    return output_grid;

}


void AddPadding_Block(Block** block,unisgned int  padding){

    Block *output_Block=(Block*)malloc(sizeof(Block));

    output_Block->depth=(*block)->depth;

    output_Block->height=(*block)->height+2*padding;
    output_Block->width=(*block)->width+2*padding;
    output_Block->matrix=(float***)malloc(output_Block->depth*sizeof(float**));


    unisgned int  counter_depth;

    for(counter_depth=0;counter_depth<output_Block->depth;counter_depth++){



            Grid* padded_grid=(Grid*)malloc(sizeof(Grid));
            padded_grid->height=(*block)->height;
            padded_grid->width=(*block)->height;

            padded_grid->grid=(*block)->matrix[counter_depth];

            *(output_Block->matrix+counter_depth)=AddPadding_Grid(&padded_grid,padding)->grid;

            free(padded_grid);


    }

    *block=output_Block;

}

float convolve_multiplication_sum(Block* block1, Block* block2){
    unisgned int  depth,width,height;

    if(block1->depth!=block2->depth || block1->depth!=block2->depth ||\
                            block1->depth!=block2->depth){
                                ERROR_DIMENSION_CONV;
                                exit(0);
                            }
    else{
        float output=0;

        for(depth=0;depth<block1->depth;depth++){

            for(height=0;height<block1->height;height++){

                for(width=0;width<block1->width;width++){
                    output+=block1->matrix[depth][height][width]*(block2->matrix[depth][height][width]);
            }
        }
    }

        return output;
    }
}



Grid* convolve(Block* block, Block* kernel, unisgned int  stride, unisgned int  padding){

    //DEBUG;
    if(!control_parity_kernel_size(kernel->height) || !control_parity_kernel_size(kernel->width)){

        ERROR_EVEN_DIMENSIONS;
        exit(0);
    }else
    if(block==NULL || kernel==NULL){

        ERROR_NULL;
        exit(0);

    }
    else
    if(block->depth!=kernel->depth){

        ERROR_DEPTH;
        exit(0);

    }
    else{

    unisgned int  height=block->height;

    AddPadding_Block(&block,padding);

    unisgned int  size_half_kernel=((kernel->height-1)/2);
    unisgned int  begin_pounisgned int _height=size_half_kernel;
    unisgned int  end_pounisgned int _height=block->height-begin_pounisgned int _height;

    unisgned int  begin_pounisgned int _width=size_half_kernel;
    unisgned int  end_pounisgned int _width=block->width-begin_pounisgned int _width;

    unisgned int  index_height_output;
    unisgned int  index_width_output;

    Grid* output_convolution_grid=(Grid*)malloc(sizeof(Grid));
    output_convolution_grid->height=(unisgned int )(end_pounisgned int _height-begin_pounisgned int _height)/stride;
    output_convolution_grid->width=(unisgned int )(end_pounisgned int _width-begin_pounisgned int _width)/stride;

    float** grid=(float**)malloc(output_convolution_grid->height*sizeof(float*));


    for(index_height_output=begin_pounisgned int _height;index_height_output<end_pounisgned int _height;index_height_output+=stride){

        float *row=(float*)malloc(output_convolution_grid->width*sizeof(float));

        for(index_width_output=begin_pounisgned int _width;index_width_output<end_pounisgned int _width;index_width_output+=stride){


            Block* extracted_block=Extract_From_Block(block,0,kernel->depth,index_height_output-size_half_kernel,\
                                                      index_height_output+size_half_kernel+1,index_width_output-size_half_kernel,\
                                                      index_width_output+size_half_kernel+1);


            *(row+(index_width_output-begin_pounisgned int _width)/stride)=convolve_multiplication_sum(extracted_block,kernel);

        }
        *(grid+(index_height_output-begin_pounisgned int _height)/stride)=row;
    }


    output_convolution_grid->grid=grid;

    return output_convolution_grid;
   }
}


//Can either be used to define the input images or the N * filters

void Convolution(Block **input, Blocks * kernels, unisgned int  stride, unisgned int  padding){


    current_Layer("Convolution");

    if(!test_Block(*input)){
        ERROR_NULL;
        exit(0);
    }
    else
    {

        Block* output=(Block*)malloc(sizeof(Block));
        output->depth=kernels->length;
        output->height=determine_size_output((*input)->height,kernels->blocks[0]->height, padding, stride);
        output->width=determine_size_output((*input)->width,kernels->blocks[0]->width, padding, stride);
        output->matrix=(float***)malloc(output->depth*sizeof(float**));


        // We have now to fill the output_matrix;

        unisgned int  index_output_depth;
        for(index_output_depth=0;index_output_depth<output->depth;index_output_depth++){


            Grid* grid=convolve(*input, kernels->blocks[index_output_depth],stride,padding);
            *(output->matrix+index_output_depth)=grid->grid;
            free(grid);

        }

        *input=output;
    }

}

float Pooling_On_Extracted_Grid(Grid* block, char* choice){
    unisgned int  width,height;

    if(choice=="max"){
        float output=0;

        for(height=0;height<block->height;height++){
            for(width=0;width<block->width;width++){

                output=max(output,block->grid[height][width]);

            }
        }

        return output;

    }else
    if(choice=="average"){

        //// Let us see
        float output=0;
        for(height=0;height<block->height;height++){
            for(width=0;width<block->width;width++){

                    output+=block->grid[height][width];

            }
        }

        return output/((block->height)*(block->width));

    }
    }


Grid* Pooling_On_Grid(Grid* grid, unisgned int  size_kernel, unisgned int  stride, unisgned int  padding,char* choice){

    if(grid==NULL){
        ERROR_NULL;
        exit(0);

    }else

    if(!control_parity_kernel_size(size_kernel)){
        ERROR_EVEN_DIMENSIONS;
        exit(0);

    }
    else {

    unisgned int  height=grid->height;

    grid=AddPadding_Grid(&grid,padding);

    // We might as well add a size_output for the width

    unisgned int  size_half_kernel=((size_kernel-1)/2);
    unisgned int  begin_pounisgned int _height=size_half_kernel;
    unisgned int  end_pounisgned int _height=grid->height-begin_pounisgned int _height;

    unisgned int  begin_pounisgned int _width=size_half_kernel;
    unisgned int  end_pounisgned int _width=grid->width-begin_pounisgned int _width;

    unisgned int  index_height_output;
    unisgned int  index_width_output;

    Grid* output_convolution_grid=(Grid*)malloc(sizeof(Grid));
    output_convolution_grid->height=(end_pounisgned int _height-begin_pounisgned int _height)/stride;
    output_convolution_grid->width=(end_pounisgned int _width-begin_pounisgned int _width)/stride;

    output_convolution_grid->grid=(float**)malloc(output_convolution_grid->height*sizeof(float*));


    for(index_height_output=begin_pounisgned int _height;index_height_output<end_pounisgned int _height;index_height_output++){
        float *row=(float*)malloc(output_convolution_grid->width*sizeof(float));

        for(index_width_output=begin_pounisgned int _width;index_width_output<end_pounisgned int _width;index_width_output++){

            Grid* extracted_grid=Extract_From_Grid(grid,index_height_output-size_half_kernel,\
                                                      index_height_output+size_half_kernel+1,index_width_output-size_half_kernel,\
                                                      index_width_output+size_half_kernel+1);


            *(row+(index_width_output-begin_pounisgned int _width)/stride)=Pooling_On_Extracted_Grid(extracted_grid,choice);

        }
        *(output_convolution_grid->grid+(index_height_output-begin_pounisgned int _height)/stride)=row;
    }



    return output_convolution_grid;

    }
}

// We will continue at this level
void Pooling(Block **input, unisgned int  size_kernel, unisgned int  stride, unisgned int  padding, char* choice){

    current_Layer("Pooling");

    if(!test_Block(*input)){
        ERROR_NULL;
        exit(0);
    }
    else
    {

    Block* output=(Block*)malloc(sizeof(Block));
    output->height=determine_size_output((*input)->height, size_kernel, padding, stride);
    output->width=determine_size_output((*input)->width, size_kernel, padding, stride);

    output->depth=(*input)->depth;
    output->matrix=(float***)malloc(output->depth*sizeof(float**));

    // We have now to fill the output_matrix;


    unisgned int  index_output_depth;
    for(index_output_depth=0;index_output_depth<output->depth;index_output_depth++){

        Grid* grid_from_current_block=(Grid*)malloc(sizeof(Grid));
        grid_from_current_block->height=(*input)->height;
        grid_from_current_block->width=(*input)->width;
        grid_from_current_block->grid=(*input)->matrix[index_output_depth];

        Grid* pooled_grid=Pooling_On_Grid(grid_from_current_block,size_kernel,stride,padding,choice);

        *(output->matrix+index_output_depth)=pooled_grid->grid;

        free(grid_from_current_block);
        free(pooled_grid);

    }

    *input=output;

    }
}


// Take a block and decrease both its width and height to 1
void Flatten(Block **input){

    //Block* block=*(input);
    current_Layer("Flatten");
    Block* block=(Block*)malloc(sizeof(Block));

    block->depth=(*input)->depth*(*input)->height*(*input)->width;
    block->width=1;
    block->height=1;

    float*** Flattened=(float***)malloc(block->depth*sizeof(float**));

    unisgned int  index_depth;
    unisgned int  index_height;
    unisgned int  index_width;

    unisgned int  counter_array_flattened=0;

    for(index_depth=0;index_depth<(*input)->depth;index_depth++){
            for(index_height=0;index_height<(*input)->height;index_height++){
                for(index_width=0;index_width<(*input)->width;index_width++){

                    float** decrease_height=(float**)malloc(sizeof(float*));
                    float* decrease_width=(float*)malloc(sizeof(float));

                    *decrease_width=(*input)->matrix[index_depth][index_height][index_width];
                    *decrease_height=decrease_width;


                    Flattened[counter_array_flattened]=decrease_height;

                    counter_array_flattened++;

            }

        }

    }

    block->matrix=Flattened;
    *input=block;

}

// Creating the Fully connected layers


void display_Block(Block* grid){

    unisgned int  dpth,row,col;

    for(dpth=0;dpth<grid->depth;dpth++){
        prunisgned int f("Level : %d\n",dpth+1);
        for(row=0;row<grid->height;row++){
            for(col=0;col<grid->width;col++){
                prunisgned int f("%.2f |", grid->matrix[dpth][row][col]);
            }
            prunisgned int f("\n");
        }
        prunisgned int f(" \n \n");
    }
    prunisgned int f("\n");
}


void display_Grid(Grid *table){
    unisgned int  row,col;

    for(row=0;row<table->height;row++){
        for(col=0;col<table->width;col++){
            prunisgned int f("%.5f |", table->grid[row][col]);
            }
        prunisgned int f("\n");
    }


}

void debug_code(){

    //Creating random input

    Block* input;
    create_Block(&input,5,10,10,"random");

    //Creating random kernels
    Blocks* kernels;
    create_Blocks(&kernels,5,5,3,3,"random");

    //Covolution Layer
    Convolution(&input,kernels,1,1);
    shape_block(input);

    //Pooling Layer
    Pooling(&input,3,2,1,"max");
    shape_block(input);

    Blocks* kernels_bis;
    create_Blocks(&kernels_bis,15,5,5,5,"random");

    //Covolution Layer
    Convolution(&input,kernels_bis,1,2);
    shape_block(input);

    //Pooling Layer
    Pooling(&input,5,1,0,"max");
    shape_block(input);


    Flatten(&input);
    shape_block(input);

    //Display Input

}


int main()
{

    //Debugging the code
    debug_code();


    return 0;
}

