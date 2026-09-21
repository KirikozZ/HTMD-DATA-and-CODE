#define NOMINMAX
#include <windows.h>
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iomanip>
using namespace std;
struct Map {
 HANDLE f,m;const char *p,*end;
 Map(const char*name){f=CreateFileA(name,GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_SEQUENTIAL_SCAN,nullptr);
 if(f==INVALID_HANDLE_VALUE)throw runtime_error("open");LARGE_INTEGER s;GetFileSizeEx(f,&s);
 m=CreateFileMappingA(f,nullptr,PAGE_READONLY,0,0,nullptr);p=(const char*)MapViewOfFile(m,FILE_MAP_READ,0,0,0);if(!p)throw runtime_error("map");end=p+s.QuadPart;}
 ~Map(){UnmapViewOfFile(p);CloseHandle(m);CloseHandle(f);}
};
struct Reader{const char*p,*end;string line(){const char*q=(const char*)memchr(p,'\n',end-p);if(!q)throw runtime_error("truncated");string s(p,q);p=q+1;return s;}
void expect(const char*s){if(line().rfind(s,0)!=0)throw runtime_error(string("header ")+s);}};
struct V{double x,y,z;};
double mic(double x,double L){return x-L*round(x/L);}
double wrap(double x,double L){return x-L*floor(x/L);}
int species(int t){return t==4?0:t==6?1:t==7?2:t==8?3:-1;}
const int B=12,NZ=176,NR=120,NH=40;
const double LO=39.803101,HI=53.043098;
struct Bin{double n[4]{},co=0,p2=0,theta=0,cn[3]{},cnlo[3]{},cnhi[3]{};};
struct Region{double n[4]{},co=0,p2=0,theta=0,cn[3]{},cnlo[3]{},cnhi[3]{};};
struct Block{int frames=0;array<Bin,NZ>bins;array<Region,3>reg;double orient[NH]{};
double rdf[3][NR]{},rdfnorm[3]{};double hydr[3][3][32]{};};
struct Episode{bool active=false,left=false;double entry=0;int entryblock=0;};
int region(double z){return z<LO?0:z>HI?2:1;}
int main(int argc,char**argv){try{
 if(argc!=4)throw runtime_error("usage trajectory water_bonds outdir");
 string out=argv[3];Map mm(argv[1]);Reader in{mm.p,mm.end};
 vector<array<int,3>> water;ifstream wfile(argv[2]);int o,h1,h2;while(wfile>>o>>h1>>h2)water.push_back({o,h1,h2});
 if(water.size()!=1872)throw runtime_error("water topology");
 vector<Block> blocks(B);vector<V>xyz;vector<int>types,seen,ions;vector<Episode>episodes;
 double L[3]{},bounds[6]{},dz=0;int N=0,frames=0;long long oldstep=-1;double time=0;
 double bondmin=1e9,bondmax=0;
 while(in.p<in.end){
  in.expect("ITEM: TIMESTEP");long long step=stoll(in.line());
  in.expect("ITEM: NUMBER OF ATOMS");int n=stoi(in.line());
  in.expect("ITEM: BOX BOUNDS pp pp pp");for(int k=0;k<3;k++){string l=in.line();double low,high;if(sscanf(l.c_str(),"%lf %lf",&low,&high)!=2)throw runtime_error("box");
  if(frames && (abs(low-bounds[2*k])>1e-8||abs(high-bounds[2*k+1])>1e-8))throw runtime_error("changed box");
  bounds[2*k]=low;bounds[2*k+1]=high;L[k]=high-low;if(abs(low)>1e-9)throw runtime_error("nonzero origin");}
  in.expect("ITEM: ATOMS id type x y z");
  if(!frames){N=n;xyz.resize(n+1);types.resize(n+1);seen.resize(n+1,-1);episodes.resize(n+1);dz=L[2]/NZ;}
  if(n!=N || (frames && step-oldstep!=10000))throw runtime_error("frame continuity");
  int b=(step-1)/10000000;if(b<0||b>=B)throw runtime_error("unexpected step");
  Block &bl=blocks[b];bl.frames++;time=step*1e-6;
  for(int i=0;i<n;i++){
   const char*e=(const char*)memchr(in.p,'\n',in.end-in.p);if(!e)throw runtime_error("atom eof");
   char*q;int id=strtol(in.p,&q,10);int ty=strtol(q,&q,10);double x=strtod(q,&q),y=strtod(q,&q),z=strtod(q,&q);
   if(id<1||id>n||seen[id]==frames||q>e||!isfinite(x+y+z))throw runtime_error("bad atom");
   if(frames&&types[id]!=ty)throw runtime_error("type changed");
   seen[id]=frames;types[id]=ty;xyz[id]={wrap(x,L[0]),wrap(y,L[1]),wrap(z,L[2])};
   if(!frames&&ty>=6&&ty<=8)ions.push_back(id);
   in.p=e+1;
  }
  double bulkO=0;
  for(auto ww:water){
   V a=xyz[ww[0]],v=xyz[ww[1]],u=xyz[ww[2]];
   V d1{mic(v.x-a.x,L[0]),mic(v.y-a.y,L[1]),mic(v.z-a.z,L[2])};
   V d2{mic(u.x-a.x,L[0]),mic(u.y-a.y,L[1]),mic(u.z-a.z,L[2])};
   double l1=sqrt(d1.x*d1.x+d1.y*d1.y+d1.z*d1.z),l2=sqrt(d2.x*d2.x+d2.y*d2.y+d2.z*d2.z);
   bondmin=min(bondmin,min(l1,l2));bondmax=max(bondmax,max(l1,l2));
   double xx=d1.x+d2.x,yy=d1.y+d2.y,zz=d1.z+d2.z;
   double co=zz/sqrt(xx*xx+yy*yy+zz*zz),p2=(3*co*co-1)/2;
   int iz=min(NZ-1,int(a.z/dz)),r=region(a.z);
   bl.bins[iz].n[0]++;bl.bins[iz].co+=co;bl.bins[iz].p2+=p2;bl.bins[iz].theta+=acos(max(-1.0,min(1.0,co)))*180/acos(-1.0);
   bl.reg[r].n[0]++;bl.reg[r].co+=co;bl.reg[r].p2+=p2;
   if(r==1)bl.orient[min(NH-1,max(0,int((co+1)*NH/2)))]++;
   if(a.z>=6&&a.z<30)bulkO++;
  }
  oldstep=step;frames++;
  if(frames%3000==0)cerr<<"frames="<<frames<<"\n";
 }
 if(frames!=12000||oldstep!=120000000)throw runtime_error("incomplete trajectory");
 if(bondmin<.95||bondmax>1.05)throw runtime_error("water bond geometry");
 ofstream prof(out+"/angle_blocks.csv");prof<<"block,frames,z_A,dz_A,water_n,cos_sum,P2_sum,theta_sum_deg\n"<<setprecision(16);
 for(int b=0;b<B;b++)for(int j=0;j<NZ;j++){auto &v=blocks[b].bins[j];prof<<b<<','<<blocks[b].frames<<','<<(j+.5)*dz<<','<<dz<<','<<v.n[0]<<','<<v.co<<','<<v.p2<<','<<v.theta<<"\n";}
 ofstream audit(out+"/parser_audit.txt");audit<<"frames="<<frames<<" last_step="<<oldstep<<" water_bond_min_A="<<bondmin<<" water_bond_max_A="<<bondmax<<" dz_A="<<dz<<"\n";
 cout<<"OK frames="<<frames<<"\n";
 }catch(exception&e){cerr<<"ERROR "<<e.what()<<"\n";return 1;}
}
