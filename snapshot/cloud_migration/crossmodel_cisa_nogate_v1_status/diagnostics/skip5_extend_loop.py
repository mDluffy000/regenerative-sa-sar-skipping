"""Original epoch loop with checkpoint/optimizer/scheduler restoration. Executed in run module globals."""
def train_resume(kind,name,m,teacher,hw,x,y,ids,valid):
    attach(m,hw,kind)
    last_path=OUT/(name+'_last.pt')
    resume=torch.load(last_path,weights_only=True,map_location='cpu') if last_path.exists() else None
    if resume is not None:
        m.load_state_dict(resume['model'])
        chosen=torch.load(OUT/(name+'_best.pt'),weights_only=True,map_location='cpu')
        bestv=chosen['validation'];bestep=chosen['epoch'];best=(bestv['accuracy'],-bestv['CE'])
        history=json.loads((OUT/(name+'_history.json')).read_text())
        start_epoch=int(resume['epoch'])
        assert len(history)==start_epoch and history[-1]['epoch']==start_epoch
        assert 0<=bestep<=start_epoch<=EPOCHS
    else:
        initial=evaluate(kind,m,x,y,valid,hw)
        best=(initial['accuracy'],-initial['CE']);bestep=0;bestv=initial;history=[];start_epoch=0
        torch.save(dict(model=m.state_dict(),epoch=0,validation=initial),OUT/(name+'_best.pt'))
    regular=[];scales=[]
    for n,p in m.named_parameters(): (scales if 'scale' in n else regular).append(p)
    opt=torch.optim.AdamW([dict(params=regular,weight_decay=.01),dict(params=scales,weight_decay=0.)],lr=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-6)
    if resume is not None:
        opt.load_state_dict(resume['optimizer'])
        assert start_epoch==5 and all(abs(g['lr']-1e-6)<1e-15 for g in opt.param_groups)
        sched=torch.optim.lr_scheduler.ConstantLR(opt,factor=1.0,total_iters=15)
        assert all(abs(g['lr']-lr)<1e-15 for g,lr in zip(opt.param_groups,sched.get_last_lr()))
    save(name+'_RESUME.json',dict(completed_epoch=start_epoch,target_epochs=EPOCHS,optimizer_restored=resume is not None,scheduler_last_epoch=sched.last_epoch,learning_rates=[g['lr'] for g in opt.param_groups],epoch_seed_rule='SEED+300+epoch; original batch order and augmentation reseeded every epoch'))
    STATE['stages'][name]=dict(epochs_finished=start_epoch,best_epoch=bestep,validation=bestv)
    if start_epoch==EPOCHS:
        status('already_complete',model=kind,branch=name,epochs=EPOCHS);return m
    bnbuf={n:v.clone() for n,v in m.state_dict().items() if n.endswith(('running_mean','running_var','num_batches_tracked'))}
    for epoch in range(start_epoch+1,EPOCHS+1):
        torch.manual_seed(SEED+300+epoch)
        order=ids[torch.randperm(len(ids)).numpy()]
        m.train();freeze_bn(m);hw.reset_stats();begin=time.monotonic();loss_sum=0;correct=0
        for pos in range(0,len(order),BATCH):
            ix=order[pos:pos+BATCH];a=prep(x[ix],kind,True);labels=y[ix].cuda()
            hw.context(ix,SEED+1000+epoch);opt.zero_grad(set_to_none=True)
            with torch.no_grad(): target=teacher(a)
            z=m(a);loss=F.cross_entropy(z,labels)+2*F.kl_div(F.log_softmax(z/2,1),F.softmax(target/2,1),reduction='batchmean')
            assert torch.isfinite(loss);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
            with torch.no_grad():
                for n,p in m.named_parameters():
                    if n.endswith(('activation_scale','weight_scale')): p.clamp_(min=1e-8)
            loss_sum+=float(loss.detach())*len(ix);correct+=int((z.detach().argmax(1)==labels).sum())
            if pos%512==0: status('training',model=kind,branch=name,epoch=epoch,epochs=EPOCHS,seen=pos+len(ix),total=len(ids),seconds=time.monotonic()-begin)
        train_hw=hw.metrics();sched.step()
        assert all(torch.equal(v,m.state_dict()[n]) for n,v in bnbuf.items())
        val=evaluate(kind,m,x,y,valid,hw);score=(val['accuracy'],-val['CE'])
        row=dict(epoch=epoch,validation=val,training_accuracy=correct/len(ids),loss=loss_sum/len(ids),training_hardware=train_hw,seconds=time.monotonic()-begin)
        history.append(row)
        if score>best:
            best=score;bestep=epoch;bestv=val
            torch.save(dict(model=m.state_dict(),epoch=epoch,validation=val),OUT/(name+'_best.pt'))
        torch.save(dict(model=m.state_dict(),epoch=epoch,optimizer=opt.state_dict(),scheduler=sched.state_dict()),OUT/(name+'_last.pt'))
        save(name+'_history.json',history)
        STATE['stages'][name]=dict(epochs_finished=epoch,best_epoch=bestep,validation=bestv)
        report();status('epoch_complete',model=kind,branch=name,epoch=epoch,validation_accuracy=val['accuracy'],best_accuracy=bestv['accuracy'])
    return m
