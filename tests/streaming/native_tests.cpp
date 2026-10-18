#include "../../firmware/streaming/protocol.hpp"
#include <cassert>
#include <vector>
#include <iostream>
#include <iterator>
struct Context{std::vector<uint8_t> out;unsigned calls=0;};
static bool forward(void*ctx,int8_t*input,float*out){auto*c=(Context*)ctx;c->calls++;out[0]=float(input[0]);out[1]=float(input[10999]);return true;}
static void send(void*ctx,const uint8_t*p,size_t n){auto*c=(Context*)ctx;c->out.insert(c->out.end(),p,p+n);}
static std::vector<uint8_t> request(uint8_t type,uint64_t session,uint32_t id,uint32_t trial,const std::vector<uint8_t>&p){std::vector<uint8_t>b(36+p.size()),out(11200);std::memcpy(b.data(),"EEG2",4);b[4]=1;b[5]=type;eeg::put64(b.data()+8,session);eeg::put32(b.data()+16,id);eeg::put32(b.data()+20,trial);b[28]=uint8_t(p.size());b[29]=uint8_t(p.size()>>8);std::memcpy(b.data()+32,p.data(),p.size());eeg::put32(b.data()+32+p.size(),eeg::crc(b.data(),32+p.size()));out.resize(eeg::encode(b.data(),b.size(),out.data(),out.size()));return out;}
int main(int argc,char**){
 alignas(16)int8_t input[11000];Context ctx;eeg::Server server(input,forward,send,&ctx);eeg::Receiver rx;
 if(argc>1){char ch;while(std::cin.get(ch)){rx.feed(uint8_t(ch),0,[&](const eeg::Frame&f){server.handle(f);});if(!ctx.out.empty()){std::cout.write((const char*)ctx.out.data(),ctx.out.size());std::cout.flush();ctx.out.clear();}}return 0;}
 assert(eeg::crc((const uint8_t*)"123456789",9)==0xcbf43926);
 auto feed=[&](const std::vector<uint8_t>&b,uint32_t time=0){for(uint8_t c:b)rx.feed(c,time,[&](const eeg::Frame&f){server.handle(f);});};
 feed(request(eeg::Hello,123,0,~0u,{}));assert(!ctx.out.empty());ctx.out.clear();
 std::vector<uint8_t> payload(11000);for(size_t i=0;i<payload.size();i++)payload[i]=uint8_t(i);
 auto good=request(eeg::Infer,123,1,0,payload);feed(good);assert(ctx.calls==1);assert(!std::memcmp(input,payload.data(),11000));ctx.out.clear();feed(good);assert(ctx.calls==1);
 feed(request(eeg::Infer,123,1,1,payload));assert(ctx.calls==1);
 feed(request(eeg::Infer,123,2,144,payload));assert(ctx.calls==1);
 feed(request(eeg::Infer,124,2,1,payload));assert(ctx.calls==1);
 feed(request(eeg::Infer,123,2,1,std::vector<uint8_t>(10999)));assert(ctx.calls==1);
 auto bad=request(eeg::Infer,123,2,1,payload);bad[100]^=1;feed(bad);assert(ctx.calls==1);
 auto partial=request(eeg::Infer,123,2,1,payload);feed(std::vector<uint8_t>(partial.begin(),partial.begin()+100),1);rx.tick(1001);feed(std::vector<uint8_t>(partial.begin()+100,partial.end()),1002);assert(ctx.calls==1);assert(rx.timeouts==1);
 feed(request(eeg::Infer,123,2,1,payload),1003);assert(ctx.calls==2);
 // Oversize and arbitrary garbage are bounded and cannot call the model.
 feed(std::vector<uint8_t>(12000,1),1004);feed({0},1004);assert(ctx.calls==2);
 for(int i=0;i<1000;i++){auto garbage=std::vector<uint8_t>(size_t(i%600),uint8_t(i%255+1));garbage.push_back(0);feed(garbage,1005);}assert(ctx.calls==2);
 for(uint32_t i=3;i<603;i++)feed(request(eeg::Infer,123,i,i%144,payload),1006);
 assert(ctx.calls==602);std::cout<<"PASS: C++ CRC, all-byte INT8, bounds, corruption, timeout, session, duplicate/conflict and 600-request reuse soak\n";
}
