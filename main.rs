use std::{env, fs::File, io::{self, Read, Write}, path::Path};

const IN: usize = 16 * 16 * 3;
const H: usize = 128;
const OUT: usize = 10;
const CLASSES: [&str; 10] = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"];

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { self.0 ^= self.0 << 7; self.0 ^= self.0 >> 9; self.0 }
    fn f(&mut self) -> f32 { ((self.next() >> 40) as f32 / 16_777_216.0) - 0.5 }
    fn usize(&mut self, n: usize) -> usize { (self.next() as usize) % n }
}

struct Model { w1: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32> }
impl Model {
    fn new() -> Self { let mut r=Rng(0x1234_5678_9abc_def0); let mut w1=vec![0.; IN*H]; let mut w2=vec![0.; H*OUT];
        let s1=(2.0/IN as f32).sqrt(); let s2=(2.0/H as f32).sqrt();
        for x in &mut w1 { *x=r.f()*s1; } for x in &mut w2 { *x=r.f()*s2; }
        Self{w1,b1:vec![0.;H],w2,b2:vec![0.;OUT]}
    }
    fn forward(&self, x:&[f32], z:&mut [f32;H], p:&mut [f32;OUT]) {
        for j in 0..H { let mut v=self.b1[j]; for i in 0..IN { v+=self.w1[j*IN+i]*x[i]; } z[j]=if v>0. {v} else {0.}; }
        let mut mx=f32::NEG_INFINITY; for k in 0..OUT { let mut v=self.b2[k]; for j in 0..H {v+=self.w2[k*H+j]*z[j];} p[k]=v; if v>mx {mx=v;} }
        let mut sum=0.; for k in 0..OUT {p[k]=(p[k]-mx).exp(); sum+=p[k];} for k in 0..OUT {p[k]/=sum;}
    }
    fn save(&self, path:&str)->io::Result<()> { let mut f=File::create(path)?; f.write_all(b"TV01")?; f.write_all(&(IN as u32).to_le_bytes())?; f.write_all(&(H as u32).to_le_bytes())?;
        for v in self.w1.iter().chain(self.b1.iter()).chain(self.w2.iter()).chain(self.b2.iter()) { f.write_all(&v.to_le_bytes())?; } Ok(()) }
    fn load(path:&str)->io::Result<Self> { let mut f=File::open(path)?; let mut h=[0;12]; f.read_exact(&mut h)?; if &h[..4]!=b"TV01" {return Err(io::Error::new(io::ErrorKind::InvalidData,"bad model"));}
        let n=(IN*H+H+H*OUT+OUT)*4; let mut b=vec![0;n]; f.read_exact(&mut b)?; let mut q=0; let mut take=|len:usize|{let mut v=Vec::with_capacity(len); for _ in 0..len {v.push(f32::from_le_bytes([b[q],b[q+1],b[q+2],b[q+3]]));q+=4;}v}; Ok(Self{w1:take(IN*H),b1:take(H),w2:take(H*OUT),b2:take(OUT)}) }
}

fn cifar_file(path:&str)->io::Result<(Vec<Vec<f32>>,Vec<u8>)> { let mut b=Vec::new(); File::open(path)?.read_to_end(&mut b)?; if b.len()%3073!=0 {return Err(io::Error::new(io::ErrorKind::InvalidData,"CIFAR file size is not a multiple of 3073"));} let n=b.len()/3073; let mut xs=Vec::with_capacity(n); let mut ys=Vec::with_capacity(n);
    for s in 0..n { let o=s*3073; ys.push(b[o]); let mut x=vec![0.;IN]; for yy in 0..16 {for xx in 0..16 {for c in 0..3 {let mut sum=0.; for dy in 0..2 {for dx in 0..2 {let src=c*1024+(yy*2+dy)*32+xx*2+dx; sum+=b[o+1+src] as f32/255.;}} x[(yy*16+xx)*3+c]=sum/4.;}}} xs.push(x); } Ok((xs,ys)) }

fn load_ppm(path:&str)->io::Result<Vec<f32>> { let mut b=Vec::new(); File::open(path)?.read_to_end(&mut b)?; let mut i=0; let tok=|b:&[u8],i:&mut usize|->io::Result<String>{while *i<b.len()&&(b[*i]==b' '||b[*i]==b'\n'||b[*i]==b'\r'||b[*i]==b'\t'){*i+=1;} if *i>=b.len(){return Err(io::Error::new(io::ErrorKind::UnexpectedEof,"PPM header"));} let s=*i; while *i<b.len()&&!b[*i].is_ascii_whitespace(){*i+=1;} Ok(String::from_utf8_lossy(&b[s..*i]).into_owned())};
    if tok(&b,&mut i)?!="P6" {return Err(io::Error::new(io::ErrorKind::InvalidData,"only binary PPM P6 is supported"));} let w:usize=tok(&b,&mut i)?.parse().unwrap(); let h:usize=tok(&b,&mut i)?.parse().unwrap(); let max: f32=tok(&b,&mut i)?.parse().unwrap(); while i<b.len()&&b[i].is_ascii_whitespace(){i+=1;} if w==0||h==0||i+3*w*h>b.len(){return Err(io::Error::new(io::ErrorKind::InvalidData,"bad PPM"));}
    let mut x=vec![0.;IN]; for yy in 0..16 {for xx in 0..16 {let sy=yy*h/16; let sx=xx*w/16; for c in 0..3 {x[(yy*16+xx)*3+c]=b[i+3*(sy*w+sx)+c] as f32/max;}}} Ok(x) }

fn train(data_dir:&str, model_path:&str, epochs:usize, limit:usize)->io::Result<()> { let mut allx=Vec::new(); let mut ally=Vec::new(); for n in 1..=5 {let p=format!("{}/data_batch_{}.bin",data_dir,n); if Path::new(&p).exists(){let (x,y)=cifar_file(&p)?; allx.extend(x);ally.extend(y);}} if allx.is_empty(){return Err(io::Error::new(io::ErrorKind::NotFound,"no data_batch_*.bin found"));} let n=limit.min(allx.len()); let mut m=if Path::new(model_path).exists(){Model::load(model_path).unwrap_or_else(|_|Model::new())}else{Model::new()}; let mut z=[0.;H]; let mut p=[0.;OUT]; let lr=0.0005; let mut order:Vec<usize>=(0..n).collect(); let mut rng=Rng(99);
    for e in 0..epochs { for i in (1..n).rev(){let j=rng.usize(i+1);order.swap(i,j);} let mut loss=0.; let mut correct=0; for &s in &order {m.forward(&allx[s],&mut z,&mut p); let y=ally[s] as usize; loss-=p[y].max(1e-9).ln(); let pred=(0..OUT).max_by(|&a,&b|p[a].total_cmp(&p[b])).unwrap(); if pred==y {correct+=1;}
            let mut dz=[0.;H]; for k in 0..OUT {let g=p[k]-(k==y) as i32 as f32; for j in 0..H {dz[j]+=g*m.w2[k*H+j]; m.w2[k*H+j]-=lr*g*z[j];} m.b2[k]-=lr*g;} for j in 0..H {if z[j]>0. {for i in 0..IN {m.w1[j*IN+i]-=lr*dz[j].clamp(-2.,2.)*allx[s][i];} m.b1[j]-=lr*dz[j].clamp(-2.,2.);}}
        } println!("epoch {:>2}/{}, loss {:.4}, accuracy {:.2}%",e+1,epochs,loss/n as f32,100.*correct as f32/n as f32); }
    m.save(model_path)?; println!("saved {} ({} parameters, {:.1} KB)",model_path,IN*H+H+H*OUT+OUT,((IN*H+H+H*OUT+OUT) as f32)*4.0/1024.); Ok(()) }

fn infer(model_path:&str,image:&str)->io::Result<()> {let m=Model::load(model_path)?; let x=load_ppm(image)?; let mut z=[0.;H];let mut p=[0.;OUT];m.forward(&x,&mut z,&mut p);let k=(0..OUT).max_by(|&a,&b|p[a].total_cmp(&p[b])).unwrap(); println!("{} ({:.2}% confidence)",CLASSES[k],100.*p[k]); Ok(())}
fn usage(){println!("TinyVision std-only Rust image system\n  train <cifar_dir> <model> [epochs] [images]\n  infer <model> <image.ppm>\n  gen-train <cifar_dir> <generator> [images]\n  generate <generator> <class 0-9> <output.ppm> [seed]\n  check <model> <image.ppm> <expected class 0-9>\n  eval <model> <test_batch.bin>\n\nCIFAR directory must contain data_batch_*.bin.");}
fn main(){let a:Vec<String>=env::args().collect(); let r=match a.get(1).map(String::as_str){Some("train") if a.len()>=4=>train(&a[2],&a[3],a.get(4).and_then(|x|x.parse().ok()).unwrap_or(8),a.get(5).and_then(|x|x.parse().ok()).unwrap_or(4000)),Some("infer") if a.len()>=4=>infer(&a[2],&a[3]),Some("gen-train") if a.len()>=4=>train_generator(&a[2],&a[3],a.get(4).and_then(|x|x.parse().ok()).unwrap_or(10000)),Some("generate") if a.len()>=5=>generate(&a[2],a[3].parse().unwrap_or(0),&a[4],a.get(5).and_then(|x|x.parse().ok()).unwrap_or(123)),Some("check") if a.len()>=5=>check_generated(&a[2],&a[3],a[4].parse().unwrap_or(0)),Some("eval") if a.len()>=4=>evaluate(&a[2],&a[3]),_=>{usage();Ok(())}}; if let Err(e)=r {eprintln!("error: {}",e);std::process::exit(1);}}


struct Generator { mean: Vec<f32>, std: Vec<f32> }
fn raw_cifar(data_dir:&str, limit:usize)->io::Result<(Vec<Vec<f32>>,Vec<u8>)> { let mut xs=Vec::new(); let mut ys=Vec::new(); for n in 1..=5 { let p=format!("{}/data_batch_{}.bin",data_dir,n); if !Path::new(&p).exists(){continue;} let mut b=Vec::new(); File::open(&p)?.read_to_end(&mut b)?; if b.len()%3073!=0{return Err(io::Error::new(io::ErrorKind::InvalidData,"bad CIFAR file"));} for s in 0..b.len()/3073 { if xs.len()>=limit {break;} let o=s*3073; ys.push(b[o]); let mut x=vec![0.;3072]; for i in 0..3072{x[i]=b[o+1+i] as f32/255.;} xs.push(x); } if xs.len()>=limit{break;} } if xs.is_empty(){return Err(io::Error::new(io::ErrorKind::NotFound,"no CIFAR batches found"));} Ok((xs,ys)) }
fn train_generator(data_dir:&str,path:&str,limit:usize)->io::Result<()> { let (xs,ys)=raw_cifar(data_dir,limit)?; let mut mean=vec![0.;10*3072]; let mut sq=vec![0.;10*3072]; let mut count=[0usize;10]; for (x,&y) in xs.iter().zip(ys.iter()){let c=y as usize; count[c]+=1; for i in 0..3072{mean[c*3072+i]+=x[i];sq[c*3072+i]+=x[i]*x[i];}} for c in 0..10 {if count[c]==0{continue;} for i in 0..3072{let m=mean[c*3072+i]/count[c] as f32; let v=(sq[c*3072+i]/count[c] as f32-m*m).max(0.00001).sqrt(); mean[c*3072+i]=m; sq[c*3072+i]=v;}} let mut f=File::create(path)?; f.write_all(b"TG01")?; for v in mean.iter().chain(sq.iter()){f.write_all(&v.to_le_bytes())?;} println!("saved generator {} using {} images; it generates class prototypes with learned pixel variation",path,xs.len()); Ok(()) }
fn load_generator(path:&str)->io::Result<Generator>{let mut f=File::open(path)?;let mut magic=[0;4];f.read_exact(&mut magic)?;if &magic!=b"TG01"{return Err(io::Error::new(io::ErrorKind::InvalidData,"bad generator"));}let mut b=vec![0;10*3072*4*2];f.read_exact(&mut b)?;let mut q=0;let mut take=|n:usize|{let mut v=Vec::with_capacity(n);for _ in 0..n{v.push(f32::from_le_bytes([b[q],b[q+1],b[q+2],b[q+3]]));q+=4;}v};Ok(Generator{mean:take(10*3072),std:take(10*3072)})}
fn normal(r:&mut Rng)->f32{let u1=((r.next()>>40) as f32/16_777_216.0).max(1e-6);let u2=(r.next()>>40) as f32/16_777_216.0;(-2.0*u1.ln()).sqrt()*(6.2831853*u2).cos()}
fn save_ppm(path:&str,x:&[f32])->io::Result<()> {let mut f=File::create(path)?;f.write_all(b"P6\n32 32\n255\n")?;let mut b=Vec::with_capacity(3072);for &v in x{b.push((v.clamp(0.,1.)*255.) as u8);}f.write_all(&b)}
fn generate(path:&str,class:usize,out:&str,seed:u64)->io::Result<()> {if class>=10{return Err(io::Error::new(io::ErrorKind::InvalidInput,"class must be 0..9"));}let g=load_generator(path)?;let mut r=Rng(seed);let mut x=vec![0.;3072];for i in 0..3072{x[i]=(g.mean[class*3072+i]+0.65*g.std[class*3072+i]*normal(&mut r)).clamp(0.,1.);}save_ppm(out,&x)?;println!("generated {} ({})",out,CLASSES[class]);Ok(())}

fn check_generated(model_path:&str,image:&str,expected:usize)->io::Result<()> {let m=Model::load(model_path)?;let x=load_ppm(image)?;let mut z=[0.;H];let mut p=[0.;OUT];m.forward(&x,&mut z,&mut p);let k=(0..OUT).max_by(|&a,&b|p[a].total_cmp(&p[b])).unwrap();println!("requested={} predicted={} confidence={:.2}% result={}",CLASSES[expected.min(9)],CLASSES[k],100.*p[k],if k==expected{"PASS"}else{"FAIL"});Ok(())}

fn evaluate(model_path:&str,test_file:&str)->io::Result<()> {let m=Model::load(model_path)?;let (xs,ys)=cifar_file(test_file)?;let mut z=[0.;H];let mut p=[0.;OUT];let mut correct=0;for (x,&y) in xs.iter().zip(ys.iter()){m.forward(x,&mut z,&mut p);let k=(0..OUT).max_by(|&a,&b|p[a].total_cmp(&p[b])).unwrap();if k==y as usize{correct+=1;}}println!("test accuracy {:.2}% ({}/{})",100.*correct as f32/xs.len() as f32,correct,xs.len());Ok(())}
