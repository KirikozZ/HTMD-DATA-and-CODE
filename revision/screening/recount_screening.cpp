#define NOMINMAX
#include <windows.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Read-only memory mapping avoids copying the 114 GB trajectory collection.
struct Mapping {
    HANDLE file, mapping;
    const char *begin, *end;
    Mapping(const char *name) {
        file = CreateFileA(name, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open trajectory");
        LARGE_INTEGER size;
        GetFileSizeEx(file, &size);
        mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        begin = static_cast<const char *>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
        if (!begin) throw std::runtime_error("Cannot map trajectory");
        end = begin + size.QuadPart;
    }
    ~Mapping() { UnmapViewOfFile(begin); CloseHandle(mapping); CloseHandle(file); }
};

struct Reader {
    const char *p, *end;
    std::string line() {
        const char *start = p;
        const char *stop = static_cast<const char *>(memchr(p, '\n', end-p));
        if (!stop) throw std::runtime_error("Truncated line");
        p = stop + 1;
        return std::string(start, stop);
    }
    void expect(const char *header) {
        if (line().find(header) != 0) throw std::runtime_error("Unexpected frame header");
    }
};

int number(const char *&p) {
    while (*p == ' ' || *p == '\t') ++p;
    int value = 0;
    while (*p >= '0' && *p <= '9') { value = value*10 + *p-'0'; ++p; }
    return value;
}
void skip(const char *&p) {
    while (*p == ' ' || *p == '\t') ++p;
    while (*p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') ++p;
}
int species(int t) { return t == 4 ? 0 : t == 6 ? 1 : t == 7 ? 2 : t == 8 ? 3 : -1; }
int side(double z, double lo, double hi) { return z < lo ? -1 : z > hi ? 1 : 0; }

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: recount trajectory output.csv");
        Mapping mapped(argv[1]);
        Reader r{mapped.begin, mapped.end};
        std::ofstream out(argv[2]);
        if (!out) throw std::runtime_error("Cannot open output");
        const std::array<std::string,4> names{{"water","Li","Cl","Mg"}};
        out << "step";
        for (const auto &s:names)
            for (const auto &v:{"plane_forward","plane_reverse","transit_forward","transit_reverse","feed","membrane","permeate","wrap_net","legacy_forward","legacy_reverse"})
                out << ',' << s << '_' << v;
        out << '\n';
        std::vector<double> previous, fixed_z, previous_legacy;
        std::vector<int> previous_type, last_side, seen;
        const double lower=39.803101, upper=53.043098;
        long long previous_step=-1, stride=-1;
        int frames=0, atoms0=0;
        double original_bounds[6]{};
        long long max_balance_error=0;
        std::array<long long,4> previous_permeate{};
        while (r.p < r.end) {
            r.expect("ITEM: TIMESTEP");
            long long step=std::stoll(r.line());
            r.expect("ITEM: NUMBER OF ATOMS");
            int atoms=std::stoi(r.line());
            r.expect("ITEM: BOX BOUNDS pp pp pp");
            double bounds[6];
            for (int k=0;k<3;++k) {
                auto line=r.line();
                if (sscanf(line.c_str(),"%lf %lf",&bounds[2*k],&bounds[2*k+1])!=2)
                    throw std::runtime_error("Bad bounds");
            }
            r.expect("ITEM: ATOMS id type x y z");
            double L=bounds[5]-bounds[4];
            if (frames==0) {
                atoms0=atoms;
                previous.resize(atoms+1); previous_legacy.resize(atoms+1); previous_type.resize(atoms+1,-1);
                fixed_z.resize(atoms+1); last_side.resize(atoms+1); seen.resize(atoms+1,-1);
                for(int k=0;k<6;++k) original_bounds[k]=bounds[k];
            } else {
                if (atoms!=atoms0 || step<=previous_step) throw std::runtime_error("Atom count or time changed");
                if (stride<0) stride=step-previous_step;
                if (step-previous_step!=stride) throw std::runtime_error("Nonuniform dump stride");
                for(int k=0;k<6;++k) if(std::abs(bounds[k]-original_bounds[k])>1e-6)
                    throw std::runtime_error("Box changed");
            }
            std::array<std::array<long long,10>,4> count{};
            for(int i=0;i<atoms;++i) {
                const char *endline=static_cast<const char *>(memchr(r.p,'\n',r.end-r.p));
                if(!endline) throw std::runtime_error("Truncated atom record");
                const char *p=r.p;
                int id=number(p), type=number(p);
                if(id<1 || id>atoms || seen[id]==frames) throw std::runtime_error("Invalid/duplicate atom ID");
                seen[id]=frames;
                if(frames && previous_type[id]!=type) throw std::runtime_error("Atom type changed");
                previous_type[id]=type;
                int s=species(type);
                if(s>=0 || type<=3) {
                    skip(p); skip(p);
                    char *number_end;
                    double z=strtod(p,&number_end);
                    if(number_end==p || number_end>endline || !std::isfinite(z)) throw std::runtime_error("Bad z");
                    if(type<=3) {
                        if(!frames) fixed_z[id]=z;
                        else if(std::abs(z-fixed_z[id])>1e-5) throw std::runtime_error("Framework z moved");
                    }
                    if(s>=0) {
                        // Match the original ML counting script: float32 z and a shared slab in adjacent frames.
                        float legacy_z=static_cast<float>(z);
                        if(frames) {
                            float legacy_old=static_cast<float>(previous_legacy[id]);
                            float left=static_cast<float>(lower), right=static_cast<float>(upper);
                            float slabmax=static_cast<float>(upper+5.0);
                            if(legacy_z>left && legacy_z<slabmax && legacy_old>left && legacy_old<slabmax) {
                                int delta=(legacy_z>right)-(legacy_old>right);
                                if(delta>0) count[s][8]++;
                                if(delta<0) count[s][9]++;
                            }
                        }
                        previous_legacy[id]=legacy_z;
                        // Normalize the small out-of-box excursions permitted by LAMMPS dumps.
                        z=bounds[4] + (z-bounds[4])-L*std::floor((z-bounds[4])/L);
                        int now=side(z,lower,upper);
                        count[s][now<0 ? 4 : now==0 ? 5 : 6]++;
                        if(frames) {
                            double old=previous[id], raw=z-old;
                            int image=raw < -L/2 ? 1 : raw > L/2 ? -1 : 0;
                            double current=z+image*L;
                            long long net=static_cast<long long>(std::floor((current-upper)/L)-std::floor((old-upper)/L));
                            if(net>0) count[s][0]+=net;
                            if(net<0) count[s][1]-=net;
                            count[s][7]+=image;
                            // A periodic reservoir crossing is not a membrane transit.
                            if(image) last_side[id]=now;
                            else if(now) {
                                if(last_side[id] && now!=last_side[id]) count[s][now>0 ? 2 : 3]++;
                                last_side[id]=now;
                            }
                        } else last_side[id]=now;
                        previous[id]=z;
                    }
                }
                r.p=endline+1;
            }
            out << step;
            for(int s=0;s<4;++s) {
                if(frames) {
                    long long residual=count[s][6]-previous_permeate[s]-(count[s][0]-count[s][1]-count[s][7]);
                    max_balance_error=std::max(max_balance_error,std::llabs(residual));
                }
                previous_permeate[s]=count[s][6];
                for(auto value:count[s]) out << ',' << value;
            }
            out << '\n';
            previous_step=step;
            ++frames;
        }
        std::cout << "frames=" << frames << " last_step=" << previous_step
                  << " stride=" << stride << " mass_balance_max_error=" << max_balance_error << std::endl;
        if(max_balance_error) throw std::runtime_error("Mass balance failed");
    } catch(const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
