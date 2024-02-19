
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
   float in[2];
   float w1[2][2];
   float hl[2];   // hdden layer
   float ho[2];   // output of hidden layer
   float w2[2][2];
   float ol[2];   // output layer
   float oo[2];   // output of output layer
   float b1[2][2];   // back propagation for hidden layer
   float b2[2];   // back propagation for output layer
   float target[2];
   float loss[2];

   lr = 0.1;
   in[0] = 0.5;
   in[1] = 0.3;
   target[0] = 1.0;
   target[1] = 0.0;

   srand((unsigned int)(time(0)));
   for(m=0; m<2; m++){
      for(n=0; n<2; n++){
         w1[m][n] = (float)(abs(rand()) / (RAND_MAX + 1.0));
         w2[m][n] = w1[m][n];
         printf("w1:%f, w2:%f\n", w1[m][n], w2[m][n]);
      }
   }

   for(ll=0; ll<10000000; ll++){
      hl[0] = (w1[0][0] * in[0]) + (w1[1][0] * in[1]);;
      hl[1] = (w1[0][1] * in[0]) + (w1[1][1] * in[1]);;
      ho[0] = sigmoid(hl[0]);   // output
      ho[1] = sigmoid(hl[1]);   // output
      //printf("%f, %f\n", hl[0], hl[1]);

      ol[0] = (w2[0][0] * ho[0]) + (w2[1][0] * ho[1]);;
      ol[1] = (w2[0][1] * ho[0]) + (w2[1][1] * ho[1]);;
      oo[0] = sigmoid(ol[0]);   // output
      oo[1] = sigmoid(ol[1]);   // output
      //printf("%f, %f\n", oo[0], oo[1]);

      if(target[0] >= 0.5)
         loss[0] = 0.5 * powf(target[0]-oo[0], 2.0);
      else
         loss[0] = -(0.5 * powf(target[0]-oo[0], 2.0));

      if(target[1] >= 0.5)
         loss[1] = 0.5 * powf(target[1]-oo[1], 2.0);
      else
         loss[1] = -(0.5 * powf(target[1]-oo[1], 2.0));

      //printf("loss:%f, %f\n", loss[0], loss[1]);

      b1[0][0] = (lr * in[0] * diffsigmoid(hl[0]) *
            ((w2[0][0] * diffsigmoid(ol[0])*loss[0])+
            ((w2[0][1] * diffsigmoid(ol[1])*loss[1]))));
      b1[0][1] = (lr * in[0] * diffsigmoid(hl[1]) *
            ((w2[1][0] * diffsigmoid(ol[0])*loss[0])+
            ((w2[1][1] * diffsigmoid(ol[1])*loss[1]))));
      b1[1][0] = (lr * in[1] * diffsigmoid(hl[0]) *
            ((w2[0][0] * diffsigmoid(ol[0])*loss[0])+
            ((w2[0][1] * diffsigmoid(ol[1])*loss[1]))));
      b1[1][1] = (lr * in[1] * diffsigmoid(hl[1]) *
            ((w2[0][1] * diffsigmoid(ol[0])*loss[0])+
            ((w2[1][1] * diffsigmoid(ol[1])*loss[1]))));

      b2[0] = (lr * ho[0] * diffsigmoid(ol[0]) * loss[0]);
      b2[1] = (lr * ho[1] * diffsigmoid(ol[1]) * loss[1]);

      w1[0][0] = w1[0][0] + b1[0][0];
      w1[0][1] = w1[0][1] + b1[0][1];
      w1[1][0] = w1[1][0] + b1[1][0];
      w1[1][1] = w1[1][1] + b1[1][1];
      w2[0][0] = w2[0][0] + b2[0];
      w2[0][1] = w2[0][1] + b2[1];
      w2[1][0] = w2[1][0] + b2[0];
      w2[1][1] = w2[1][1] + b2[1];
   }
   printf("w1:%f, %f, w2:%f, %f, output:%f, %f\n", w1[0][0], w1[0][1], w2[0][0], w2[0][1], oo[0], oo[1]);
   printf("loss:%f, %f\n", loss[0], loss[1]);

}
