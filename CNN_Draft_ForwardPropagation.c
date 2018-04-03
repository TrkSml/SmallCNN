#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define UPPER_BOUND 10.0
#define GEN_RANDOM_SEED srand(time(NULL))
#define DEBUG printf("debug !")
/*
typedef struct{
   int index_width;
   int index_height;
   int value;
} Block ;
*/

typedef struct {
    int depth;
    int width;
    int height;
    float*** matrix;
} Block;


float generate_random(){
    return ((float)rand())/((float)RAND_MAX) * UPPER_BOUND;

}

void create_Random_Block(Block** grid, int input_depth, int input_height, int input_width){

    *grid=(Block*)malloc(sizeof(Block));
    (*grid)->height=input_height;
    (*grid)->width=input_width;
    (*grid)->depth=input_depth;

    float ***image=malloc(input_depth*sizeof(float**));
    int counter_depth;

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
    create_Random_Block(&layer,3,10,5);
    layer1=Extract_From_Block(layer,0,3,0,3,0,3);
    display_Block(layer);
    display_Block(layer1);


    return 0;
}
