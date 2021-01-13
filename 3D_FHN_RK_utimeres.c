#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.1415926535897

int dim_x=256,dim_z;
const double mu=1;


typedef struct node {
	short int x,y,z;
	float sigma;
	struct node * next;
} node_t;

void push(node_t ** head, node_t A) {
	node_t * new_node;
	new_node = malloc(sizeof(node_t));

	new_node->x = A.x;
	new_node->y = A.y;
	new_node->z = A.z;
	new_node->sigma=A.sigma;
	new_node->next = *head;
	*head = new_node;
}


int main(int argc, char** argv)
{	
	printf("%f\n",atof(argv[5]));
	printf("0");
	if(argc!=7)
	{
		printf("Wrong Input!\nUsage: ./3D_FHN input_file z_dimention seed output_directory phi\n");
		exit(1);
	}
	dim_z=atoi(argv[2]);
	node_t **ConMatrix=malloc(dim_x*dim_x*dim_z*sizeof(node_t));
	FILE *fp,*fout;
	fp=fopen(argv[1],"r");
	if (!fp)
	{	
		printf("Input File not Found!\n");
		exit(1);
	}
	short int i,j,k,r;
	int tt=0;

	char str[100];
	//sprintf(str,"./FHN_%s_%d_%s/data/%05d.dat",argv[4],atoi(argv[3]),argv[5],tt);
	sprintf(str,"./%s/data/%05d.dat",argv[6],tt);
	printf(".%s\n",argv[6]);
	fout=fopen(str,"w");
	//CONECTIVITY MATRIX SETUP
	for(i=0;i<dim_x;i++)
	{
		for(j=0;j<dim_x;j++)
		{
			for(k=0;k<dim_z;k++)
			{
				ConMatrix[i*dim_x*dim_z + j*dim_z + k]=NULL;
			}
		}
	}
	printf("..");
	r=1;
	while(r>0)
	{
		node_t A;
		r=fscanf(fp, "%hd", &i);
		r=fscanf(fp, "%hd", &j);
		r=fscanf(fp, "%hd", &k);
		r=fscanf(fp, "%hd", &A.x);
		r=fscanf(fp, "%hd", &A.y);
		r=fscanf(fp, "%hd", &A.z);
		r=fscanf(fp, "%f",&A.sigma);
		push(&ConMatrix[i*dim_x*dim_z + j*dim_z + k],A);
	}
	printf("Initialization Completed.\n");


	//INITIALIZATION OF X,Y
	const float alpha=0.5,epsilon=0.05;
	double *X,*X_,*Y,*Y_,dt=0.01;
	double *k1x,*k1y;
	int *Omega;
	double phi=atof(argv[5]);
	double bxx=cos(phi);
	double bxy=sin(phi);
	double byx=-sin(phi);
	double byy=cos(phi);
	int ii;
	X=malloc(dim_x*dim_x*dim_z*sizeof(double));
	Y=malloc(dim_x*dim_x*dim_z*sizeof(double));

	Omega=malloc(dim_x*dim_x*dim_z*sizeof(int));

	srand(atoi(argv[3]));
	for(i=0;i<dim_x;i++)
	{
		for(j=0;j<dim_x;j++)
		{
			for(k=0;k<dim_z;k++)
			{
				if (ConMatrix[i*dim_x*dim_z + j*dim_z + k]!=NULL)
				{   
					X[i*dim_x*dim_z + j*dim_z + k]=1.9*rand()/RAND_MAX-0.9;
					Y[i*dim_x*dim_z + j*dim_z + k]=rand()/RAND_MAX-0.4;
				}
				else
				{   
					X[i*dim_x*dim_z + j*dim_z + k]=-10;
					Y[i*dim_x*dim_z + j*dim_z + k]=-10;
				}
				Omega[i*dim_x*dim_z + j*dim_z + k]=0;

			}
		}
	}
	printf("X,Y initialization completed\n");
	//MAIN PROGRAM
	X_=malloc(dim_x*dim_x*dim_z*sizeof(double));
	Y_=malloc(dim_x*dim_x*dim_z*sizeof(double));

	k1x=malloc(dim_x*dim_x*dim_z*sizeof(double));
	k1y=malloc(dim_x*dim_x*dim_z*sizeof(double));
	
	printf("X_,Y_ allocation completed\n");
	for(tt=0;tt<=21000;tt++)
	{

		node_t *A;
		#pragma omp parallel for schedule(dynamic)  private(ii,A) 
		for(ii=0; ii<dim_x*dim_x*dim_z; ii++)
		{
			int count,pos;
			A=ConMatrix[ii];
						
			k1x[ii]=0;
			k1y[ii]=0;
			
			if(A!=NULL)
			{

				double Sx=0,Sy=0;
				count=0;
				while(A!=NULL)
				{
					pos=A->x*dim_x*dim_z + A->y*dim_z + A->z;
					Sx+=A->sigma*(bxx*(X[ii] - X[pos]) + bxy*(Y[ii] - Y[pos]));
					Sy+=A->sigma*(byx*(X[ii] - X[pos]) + byy*(Y[ii] - Y[pos]));
					count++;
					A=A->next;
				}				
				k1x[ii]=dt/2* (X[ii]-X[ii]*X[ii]*X[ii]/3 -Y[ii] +1.0/count*Sx)/epsilon;
				k1y[ii]=dt/2* (X[ii]+alpha +1.0/count*Sy);
			}
		}

		#pragma omp parallel for schedule(dynamic)  private(ii,A) 
                for(ii=0; ii<dim_x*dim_x*dim_z; ii++)
                {
                        int count,pos;
			A=ConMatrix[ii];
                                                
                        X_[ii]=0;
                        Y_[ii]=0;
                                                
			if(A!=NULL)
                        {

                                double Sx=0,Sy=0;
                                count=0;
				while(A!=NULL)
                                {
                                        pos=A->x*dim_x*dim_z + A->y*dim_z + A->z;
                                        Sx+=A->sigma*(bxx*(X[ii]+k1x[ii] - X[pos]-k1x[pos]) + bxy*(Y[ii]+k1y[ii] - Y[pos]-k1y[pos]));
                                        Sy+=A->sigma*(byx*(X[ii]+k1x[ii] - X[pos]-k1x[pos]) + byy*(Y[ii]+k1y[ii] - Y[pos]-k1y[pos]));
                                        count++;
                                        A=A->next;
                                }
				X_[ii]=dt* (X[ii]+k1x[ii] -powf(X[ii]+k1x[ii],3.0)/3 -Y[ii]-k1y[ii] +1.0/count*Sx)/epsilon;
                                Y_[ii]=dt* (X[ii]+k1x[ii] +alpha +1.0/count*Sy);
			}	
			double h=X[ii];
			X[ii]=X[ii] + X_[ii];
			Y[ii]=Y[ii] + Y_[ii];
			if(tt>1000 && h<0 && X[ii]>0) Omega[ii]+=1;
			
		}
		
		
		if(tt>20000)
		{
			if (fout!=NULL) fclose(fout);
			sprintf(str,"./%s/data/%05d.dat",argv[6],tt);
			fout=fopen(str,"w");
			for(i=0;i<dim_x;i++)
			{
				for(j=0;j<dim_x;j++)
				{
					for(k=0;k<dim_z;k++)
					{
						if (X[i*dim_x*dim_z + j*dim_z + k] !=-10)
							fprintf(fout,"%d\t%d\t%d\t%lf\t%lf\t%lf\n",i,j,k,X[i*dim_x*dim_z + j*dim_z + k],Omega[i*dim_x*dim_z + j*dim_z + k]*2.0*PI/(tt/100.0-10.0),Y[i*dim_x*dim_z + j*dim_z + k]);
					}
				}
			}
		}
		
	}
	printf("%s DONE\n",argv[4]);
	return 0;
}
