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
   float out;   // output of output layer
   float ol;   // output layer
   float target;
   float loss;

   lr = 0.1;
   in = 0.5;
   target = 1.0;

   srand((unsigned int)(time(0)));
   w1 = (float)(abs(rand()) / (RAND_MAX + 1.0));
   printf("w1:%f\n", w1);

   for(ll=0; ll<100000000; ll++){
      ol = w1 * in;
      out = sigmoid(ol);   // output
      //printf("%f\n", out);

      if(target >= 0.5)
         loss = 0.5 * powf(target-out, 2.0);
      else
         loss = -(0.5 * powf(target-out, 2.0));

      //printf("loss:%f\n", loss);

 
      w1 = w1 + (lr * in * diffsigmoid(ol) * loss);
   }
   printf("w1:%f, output:%f\n", w1, out);
   printf("loss:%f\n", loss);

}
