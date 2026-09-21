"""Extract each water oxygen's two bonded hydrogens from supplied initialization files."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    for folder in sorted((ROOT.parent/'simulation/optimized').iterdir()):
        if not folder.is_dir():continue
        section='';sections={}
        for line in (folder/'TbHz.data').read_text().splitlines():
            text=line.split('#')[0].strip()
            if not text:continue
            if text[0].isalpha():section=text;sections.setdefault(section,[])
            elif section:sections[section].append(text.split())
        types={int(x[0]):int(x[2]) for x in sections['Atoms']}
        waters={i:[] for i,t in types.items() if t==4}
        for row in sections['Bonds']:
            i,j=map(int,row[2:4])
            if types[i]==4 and types[j]==5:waters[i].append(j)
            if types[j]==4 and types[i]==5:waters[j].append(i)
        assert len(waters)==1872 and all(len(h)==2 for h in waters.values())
        out=ROOT/'raw'/folder.name;out.mkdir(parents=True,exist_ok=True)
        (out/'water_bonds.txt').write_text('\n'.join(f'{o} {h[0]} {h[1]}' for o,h in waters.items())+'\n')
    print('Prepared water topology for seven systems.')
if __name__=='__main__':main()
