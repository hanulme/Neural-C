
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

float sigmoid(float x){
   return 1 / ( 1 + expf(-x));
}

float diffsigmoid(float x){
   float y;
   y = 1 / (1 + expf(-x));
   return (y * (1 - y));
}


void main(){
   int m, n;
   int ll;   // lerning loop
   float lr;   // learning rate
   float in;
   float w1;
   float hl;   // hdden layer
   float ho;   // output of hidden layer
   float w2;
   float ol;   // output layer
   float oo;   // output of output layer
   float bh;   // back propagation for hidden layer
   float target;
   float loss;

   lr = 0.1;
   in = 0.5;
   target = 1.0;

   srand((unsigned int)(time(0)));
   w1 = (float)(abs(rand()) / (RAND_MAX + 1.0));
   w2 = w1;
   printf("w1:%f, w2:%f\n", w1, w2);

   for(ll=0; ll<1000000; ll++){
      hl = w1 * in;
      ho = sigmoid(hl);   // output
      //printf("%f\n", hl);

      ol = w2 * ho;
      oo = sigmoid(ol);   // output
      //printf("%f\n", oo);

      if(target >= 0.5)
         loss = 0.5 * powf(target-oo, 2.0);
      else
         loss = -(0.5 * powf(target-oo, 2.0));

      //printf("loss:%f\n", loss);

      //if(target >= 0.5){
         w1 = w1 + (lr * in * diffsigmoid(hl) * (w2 * diffsigmoid(ol)*loss));
         w2 = w2 + (lr * ho * diffsigmoid(ol) * loss);
      //}
      //else{
      //   w1 = w1 - (lr * in * diffsigmoid(hl) * (w2 * diffsigmoid(ol)*loss));
      //   w2 = w2 - (lr * ho * diffsigmoid(ol) * loss);
      //}
   }
   printf("w1:%f, w2:%f, output:%f\n", w1, w2, oo);
   printf("loss:%f\n", loss);

}
