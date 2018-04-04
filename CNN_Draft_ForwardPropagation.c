#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define UPPER_BOUND 10.0
#define GEN_RANDOM_SEED srand(time(NULL))
#define DEBUG printf("debug !")
#define ERROR_DIMENSION printf("Dimension conflict: Please review the dimensions of the convolved arrays!")
#define ERROR_DEPTH printf("Cannot perform convolution: Please make sure the kernel and the block have the same depth.")

typedef struct {
    int depth;
    int width;
    int height;
    float*** matrix;
} Block;


float generate_random(){
    return ((float)rand())/((float)RAND_MAX) * UPPER_BOUND;

}

void create_Block(Block** grid, int input_depth, int input_height, int input_width,char* choice){

    *grid=(Block*)malloc(sizeof(Block));
    (*grid)->height=input_height;
    (*grid)->width=input_width;
    (*grid)->depth=input_depth;

    float ***image=malloc(input_depth*sizeof(float**));
    int counter_depth;

    if(choice=="random"){

    for(counter_depth=0;counter_depth<input_depth;counter_depth++){

        float **data_matrix=(float**)malloc(input_height*sizeof(float*));
        int counter_height;
        for(counter_height=0;counter_height<input_height;counter_height++){
            int counter_width;
            float* row=(float*)malloc(input_width*(sizeof(float)));
            for(counter_width=0;counter_width<input_width;counter_width++){
                        *(row+counter_width)=(float)generate_random();
                }
            *(data_matrix+counter_height)=row;
        }
        *(image+counter_depth)=data_matrix;
    }

    (*grid)->matrix=image;

    }
    else if(choice=="zeros"){

        for(counter_depth=0;counter_depth<input_depth;counter_depth++){
        float **data_matrix=(float**)malloc(input_height*sizeof(float*));
        int counter_height;
        for(counter_height=0;counter_height<input_height;counter_height++){
            int counter_width;
            float* row=(float*)malloc(input_width*(sizeof(float)));
            for(counter_width=0;counter_width<input_width;counter_width++){
                        *(row+counter_width)=0.0;
                }
            *(data_matrix+counter_height)=row;
        }
        *(image+counter_depth)=data_matrix;
    }

    (*grid)->matrix=image;

    }

}


Block* Extract_From_Block(Block* grid,\
                          int begin_input_depth, int end_input_depth,\
                          int begin_input_height, int end_input_height,\
                          int begin_input_width, int end_input_width){


    Block* output_grid=(Block*)malloc(sizeof(Block));

    int size_depth=end_input_depth-begin_input_depth;
    int size_height=end_input_height-begin_input_height;
    int size_width=end_input_width-begin_input_width;

    output_grid->depth=size_depth;
    output_grid->width=size_width;
    output_grid->height=size_height;

    float ***image=malloc(size_depth*sizeof(float**));
    float ***copy_image=grid->matrix;

    int counter_depth;
    int new_counter_depth=0;

    for(counter_depth=begin_input_depth;counter_depth<end_input_depth;counter_depth++){
        ;
        float **data_matrix=(float**)malloc(size_height*sizeof(float*));

        int counter_height;
        int new_counter_height=0;

        for(counter_height=begin_input_height;counter_height<end_input_height;counter_height++){

            int counter_width;
            int new_counter_width=0;
            float* row=(float*)malloc(size_width*(sizeof(float)));

            for(counter_width=begin_input_width;counter_width<end_input_width;counter_width++){
                        *(row+new_counter_width)=copy_image[counter_depth][counter_height][counter_width];
                        new_counter_width++;

                }

            data_matrix[new_counter_height]=row;
            new_counter_height++;
        }
        image[new_counter_depth]=data_matrix;
        new_counter_depth++;
    }

    output_grid->matrix=image;

    return output_grid;

}

Block* AddPadding(Block** block,int padding){

    Block *output_Block;

    int depth=(*block)->depth;
    int height=(*block)->height;
    int width=(*block)->width;
    float ***block_matrix=(*block)->matrix;

    create_Block(&output_Block,depth,height+2*padding,width+2*padding,"zeros");


    float*** output_Block_matrix=output_Block->matrix;


    int counter_depth;
    int counter_width;
    int counter_height;

    for(counter_depth=0;counter_depth<depth;counter_depth++){
        for(counter_height=padding;counter_height<padding+height;counter_height++){
            for(counter_width=padding;counter_width<padding+width;counter_width++){

                output_Block_matrix[counter_depth][counter_height][counter_width]=\
                    block_matrix[counter_depth][counter_height-padding][counter_width-padding];

            }
        }
    }

    return output_Block;

}

float convolve_multiplication_sum(Block* block1, Block* block2){
    int depth,width,height;
    if(block1->depth!=block2->depth || block1->depth!=block2->depth ||\
                            block1->depth!=block2->depth){
                                ERROR_DIMENSION;
                                exit(0);
                            }
    else{
        float output=0;
        for(depth=0;depth<block1->depth;depth++){
            for(height=0;height<block1->height;height++){
                for(width=0;width<block1->width;width++){
                    output+=block1->matrix[depth][height][width]\
                        *block2->matrix[depth][height][width];
            }
        }
    }

        return output;
    }
}

float ** convolve(Block* block, Block* kernel, int stride, int padding){

    if(block->depth!=kernel->depth){

        ERROR_DEPTH;
        exit(0);
    }else{

    int height=block->height;

    AddPadding(&block,padding);
    int size_output=(int)(floor((height-kernel->height+2*padding)/stride))+1;
    float output_convolution[size_output][size_output];

    int size_half_kernel=(int)((kernel->height-1)/2);
    int begin_point_height=size_half_kernel;
    int end_point_height=block->height-begin_point_height+1;

    int begin_point_width=begin_point_height;
    int end_point_width=block->width-begin_point_width+1;

    int index_height_output;
    int index_width_output;

    for(index_height_output=begin_point_height;index_height_output<end_point_height;index_height_output++){
        for(index_width_output=begin_point_width;index_width_output<size_output;begin_point_width++){
            Block* extracted_block=Extract_From_Block(block,0,kernel->depth,index_height_output-size_half_kernel,\
                                                      index_height_output+size_half_kernel,index_width_output-size_half_kernel,\
                                                      index_width_output+size_half_kernel);
            output_convolution[index_height_output-begin_point_height][index_width_output-begin_point_width]=\
                                convolve_multiplication_sum(extracted_block,kernel);
        }
    }

    return output_convolution;
   }

}



void display_Block(Block* grid){

    int dpth,row,col;

    for(dpth=0;dpth<grid->depth;dpth++){
        printf("Level : %d\n",dpth+1);
        for(row=0;row<grid->height;row++){
            for(col=0;col<grid->width;col++){
                printf("%.2f |", grid->matrix[dpth][row][col]);
            }
            printf("\n");
        }
        printf(" \n \n");
    }
    printf("\n");
}


int main()
{


    Block* layer;
    Block* layer1;
    create_Block(&layer,3,10,5,"random");
    create_Block(&layer1,3,2,2,"random");

    float **convolved;
    cnvolved=convolve(layer,layer1,1,1);
    /************ Extra *********/
    /*
    layer1=Extract_From_Block(layer,0,3,0,3,0,3);
    layer1=AddPadding(&layer,1);
    display_Block(layer1);
    */

    return 0;
    
}
