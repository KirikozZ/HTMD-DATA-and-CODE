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
struct Bin{double n[4]{},co=0,p2=0,cn[3]{},cnlo[3]{},cnhi[3]{};};
struct Region{double n[4]{},co=0,p2=0,cn[3]{},cnlo[3]{},cnhi[3]{};};
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
 ofstream ev(out+"/residence_episodes.csv");ev<<"species,atom_id,entry_ns,exit_ns,duration_ns,left_censored,right_censored,entry_block\n";ev<<setprecision(12);
 auto emit=[&](int id,int s,Episode&e,double t,bool right){ev<<s<<','<<id<<','<<e.entry<<','<<t<<','<<t-e.entry<<','<<e.left<<','<<right<<','<<e.entryblock<<"\n";};
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
   bl.bins[iz].n[0]++;bl.bins[iz].co+=co;bl.bins[iz].p2+=p2;
   bl.reg[r].n[0]++;bl.reg[r].co+=co;bl.reg[r].p2+=p2;
   if(r==1)bl.orient[min(NH-1,max(0,int((co+1)*NH/2)))]++;
   if(a.z>=6&&a.z<30)bulkO++;
  }
  int nx=int(L[0]/6),ny=int(L[1]/6),nz=int(L[2]/6);
  vector<vector<int>>cells(nx*ny*nz);
  for(auto ww:water){V a=xyz[ww[0]];int ix=min(nx-1,int(a.x/L[0]*nx)),iy=min(ny-1,int(a.y/L[1]*ny)),iz=min(nz-1,int(a.z/L[2]*nz));cells[(ix*ny+iy)*nz+iz].push_back(ww[0]);}
  double rhoO=bulkO/(L[0]*L[1]*24);
  for(int id:ions){
   int s=species(types[id])-1;V a=xyz[id];int iz=min(NZ-1,int(a.z/dz)),r=region(a.z);
   bl.bins[iz].n[s+1]++;bl.reg[r].n[s+1]++;
   bool bulk=a.z>=6 && a.z<30;
   if(bulk)bl.rdfnorm[s]+=rhoO;
   const double rc[3]={2.75,3.85,2.65};
   int cn=0,cnlo=0,cnhi=0;
   double r2=rc[s]*rc[s],lo2=(rc[s]-.1)*(rc[s]-.1),hi2=(rc[s]+.1)*(rc[s]+.1);
   int cx=min(nx-1,int(a.x/L[0]*nx)),cy=min(ny-1,int(a.y/L[1]*ny)),cz=min(nz-1,int(a.z/L[2]*nz));
   for(int dx=-1;dx<=1;dx++)for(int dy=-1;dy<=1;dy++)for(int ddz=-1;ddz<=1;ddz++){
    int k=(((cx+dx+nx)%nx)*ny+(cy+dy+ny)%ny)*nz+(cz+ddz+nz)%nz;
    for(int oid:cells[k]){V q=xyz[oid];double x=mic(q.x-a.x,L[0]),y=mic(q.y-a.y,L[1]),z=mic(q.z-a.z,L[2]);double rr=x*x+y*y+z*z;
     cn+=rr<=r2;cnlo+=rr<=lo2;cnhi+=rr<=hi2;
     if(bulk&&rr<36)bl.rdf[s][min(NR-1,int(sqrt(rr)/.05))]++;
    }
   }
   if(cn>=32)throw runtime_error("CN histogram overflow");
   bl.bins[iz].cn[s]+=cn;bl.bins[iz].cnlo[s]+=cnlo;bl.bins[iz].cnhi[s]+=cnhi;
   bl.reg[r].cn[s]+=cn;bl.reg[r].cnlo[s]+=cnlo;bl.reg[r].cnhi[s]+=cnhi;
   bl.hydr[r][s][cn]++;
  }
  for(int id=1;id<=N;id++){
   int s=species(types[id]);if(s<0)continue;
   bool now=region(xyz[id].z)==1;Episode&e=episodes[id];
   if(now&&!e.active){e.active=true;e.entry=time;e.entryblock=b;e.left=frames==0;}
   if(!now&&e.active){emit(id,s,e,time,false);e.active=false;}
  }
  oldstep=step;frames++;
  if(frames%3000==0)cerr<<"frames="<<frames<<"\n";
 }
 for(int id=1;id<=N;id++)if(episodes[id].active)emit(id,species(types[id]),episodes[id],time,true);
 if(frames!=12000||oldstep!=120000000)throw runtime_error("incomplete trajectory");
 if(bondmin<.95||bondmax>1.05)throw runtime_error("water bonds geometry");
 ofstream prof(out+"/profiles_blocks.csv");prof<<"block,frames,z_A,dz_A,water_n,Li_n,Cl_n,Mg_n,cos_sum,P2_sum,Li_CN_sum,Cl_CN_sum,Mg_CN_sum,Li_CNlo_sum,Cl_CNlo_sum,Mg_CNlo_sum,Li_CNhi_sum,Cl_CNhi_sum,Mg_CNhi_sum\n";prof<<setprecision(14);
 ofstream reg(out+"/regions_blocks.csv");reg<<"block,frames,region,water_n,Li_n,Cl_n,Mg_n,cos_sum,P2_sum,Li_CN_sum,Cl_CN_sum,Mg_CN_sum,Li_CNlo_sum,Cl_CNlo_sum,Mg_CNlo_sum,Li_CNhi_sum,Cl_CNhi_sum,Mg_CNhi_sum\n";reg<<setprecision(14);
 ofstream ori(out+"/orientation_blocks.csv");ori<<"block,cos_theta,count\n";
 ofstream rdf(out+"/rdf_blocks.csv");rdf<<"block,species,r_A,pair_count,normalizer\n";rdf<<setprecision(14);
 ofstream hyd(out+"/hydration_hist_blocks.csv");hyd<<"block,region,species,CN,count\n";
 for(int b=0;b<B;b++){auto &bb=blocks[b];
  for(int iz=0;iz<NZ;iz++){auto &v=bb.bins[iz];prof<<b<<','<<bb.frames<<','<<(iz+.5)*dz<<','<<dz;for(double a:v.n)prof<<','<<a;prof<<','<<v.co<<','<<v.p2;for(auto arr:{v.cn,v.cnlo,v.cnhi})for(int i=0;i<3;i++)prof<<','<<arr[i];prof<<"\n";}
  for(int j=0;j<3;j++){auto &v=bb.reg[j];reg<<b<<','<<bb.frames<<','<<j;for(double a:v.n)reg<<','<<a;reg<<','<<v.co<<','<<v.p2;for(auto arr:{v.cn,v.cnlo,v.cnhi})for(int i=0;i<3;i++)reg<<','<<arr[i];reg<<"\n";}
  for(int j=0;j<NH;j++)ori<<b<<','<<-1+(j+.5)*2/NH<<','<<bb.orient[j]<<"\n";
  for(int s=0;s<3;s++)for(int j=0;j<NR;j++){double shell=4*acos(-1)/3*(pow((j+1)*.05,3)-pow(j*.05,3));rdf<<b<<','<<s<<','<<(j+.5)*.05<<','<<bb.rdf[s][j]<<','<<bb.rdfnorm[s]*shell<<"\n";}
  for(int r=0;r<3;r++)for(int s=0;s<3;s++)for(int c=0;c<32;c++)hyd<<b<<','<<r<<','<<s<<','<<c<<','<<bb.hydr[r][s][c]<<"\n";
 }
 ofstream audit(out+"/parser_audit.txt");audit<<"frames="<<frames<<" last_step="<<oldstep<<" water_bond_min_A="<<bondmin<<" water_bond_max_A="<<bondmax<<" dz_A="<<dz<<"\n";
 cout<<"OK frames="<<frames<<" water_bonds="<<bondmin<<","<<bondmax<<"\n";
 }catch(exception&e){cerr<<"ERROR "<<e.what()<<"\n";return 1;}
}

