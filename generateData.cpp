#include<cstdio>
#include<algorithm>
#include<ctime>
using namespace std;

int main(){
	int T=100;//The number of cases
	int n=100000;//The number of points
	int m=100;//The number of queries 
	int d=8;//The number of dimensions
	const int mod=1e9;
	srand((int)time(0));
	char filename[100];
	for (int t=1;t<=T;t++)
	{
	  sprintf(filename,"./data/%d.txt",t);
	  freopen(filename,"w",stdout);
	  printf("%d %d %d\n",n,m,d);
	  for (int i=1;i<=n;i++)
	    for (int j=1;j<=d;j++)
	    {
	      long long num=((long long)rand()*RAND_MAX+rand())%mod;
	      if (j==d)
		    printf("%lld\n",num);
	      else
	        printf("%lld ",num);
	    }
	  for (int i=1;i<=m;i++)
	    for (int j=1;j<=d;j++)
	    {
	      long long num=((long long)rand()*RAND_MAX+rand())%mod;
	      if (j==d)
		    printf("%lld\n",num);
	      else
	        printf("%lld ",num);
	    }
	}
	return 0;
}
