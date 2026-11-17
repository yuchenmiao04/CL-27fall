"""CPU audit of a finalized, root-selected CIFAR development backup only.

Never calls the formal guard/facts/data checker: development uses sparse fitting
IDs after a train holdout. Never selects a winner, trains, evaluates accuracy,
loads CUDA, downloads weights or writes a checkpoint. All results go to stdout.
"""
import argparse,copy,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
TRIALS=('cifar_current_bounded64_full_v4','cifar_current_mb2_full_v5')
POLICY='canonical_loader_isolated_v1'


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','state-loader','native-auditor','checkpoint-backup','metadata-dir','pretrained-npz'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--selected-trial',choices=TRIALS,required=True)
    p.add_argument('--runtime-cwd',type=Path)
    p.add_argument('--confirmed-completed-development-backup',action='store_true',required=True)
    a=p.parse_args();began=time.perf_counter();os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
    repo=a.repo.resolve(strict=True);cwd=(a.runtime_cwd or repo).resolve(strict=True)
    backup=a.checkpoint_backup.resolve(strict=True);meta=a.metadata_dir.resolve(strict=True)
    loader=a.state_loader.resolve(strict=True);auditor=a.native_auditor.resolve(strict=True)
    npz=Path(os.path.abspath(a.pretrained_npz))
    auditor_sha=hashlib.sha256(auditor.read_bytes()).hexdigest();loader_sha=hashlib.sha256(loader.read_bytes()).hexdigest()
    native=module_at('trusted_development_native_auditor',auditor);native.REPO=repo;need=native.need
    need(native.sha(backup)==a.expected_sha256,'Finalized development backup SHA mismatch')
    need(npz.is_file()and native.sha(npz)==native.WEIGHT_SHA,'Existing native-cache NPZ missing or different')
    values={n:native.load_json(meta/(n+'.json'))for n in ('config','results','data_manifest')}
    cfg,result,saved=(values[n]for n in ('config','results','data_manifest'))
    need(Path(cfg['output_dir']).name==a.selected_trial,'Metadata does not identify the selected development trial')
    need(cfg['model_name']=='current_adapted'and cfg['dataset']=='cifar224','Not the CIFAR fixed-library candidate')
    need(cfg['phase']==result['phase']=='development'and result['status']=='complete','Need completed development output')
    need(cfg['seed']==1993 and cfg['shuffle']and cfg['validation_only']is True,'Wrong actual development seed/mode')
    need(cfg.get('max_tasks')in(None,10)and cfg.get('validation_max_per_class')is None,'Truncated/capped development')
    need(cfg.get('validation_fraction',.1)==.1 and cfg.get('split_seed',0)==0,'Changed internal holdout identity')
    need(cfg['epochs']==5 and cfg['init_cls']==cfg['increment']==10,'Wrong epochs/class split')
    need(not cfg.get('smoke',False)and cfg['pretrained']and cfg['backbone_type']=='vit_base_patch16_224','Wrong backbone')
    expected=(4,('modules','modules','backbone','modules'))if a.selected_trial==TRIALS[0]else(2,('modules','backbone'))
    need((cfg['outer_updates'],tuple(cfg.get('schedule',expected[1])))==expected,'Selected phase schedule differs')
    need(cfg['episode_replay_batch_size']==64 and cfg['memory_size']==2000 and cfg['fixed_memory']is False,'Wrong memory budget')
    need(saved['phase']=='development'and saved['validation_only']and saved['official_test_read']is False,'Not train-only holdout evidence')
    need((saved['train_count'],saved['evaluation_count'],saved['evaluation_source'])==(45000,5000,'training_holdout'),'Wrong development sizes')
    sys.path.insert(0,str(repo));os.chdir(cwd);comparator=module_at('independent_dev_state_loader',loader)
    sys.path=[str(repo)]+[v for v in sys.path if v!=str(repo)]
    import numpy as np
    import torch
    torch.set_num_threads(4);state,digest=comparator.load_trusted_checkpoint(backup)
    need(digest==a.expected_sha256 and state['version']==1 and state['random_stream_policy']==POLICY,'Loaded bytes/version/RNG policy differ')
    finite={n:native.finite_scan(v,np,torch)for n,v in dict(values,checkpoint=state).items()}
    from cl27.runner import configuration_fingerprint,summarize
    need(state['config_fingerprint']==result['config_fingerprint']==configuration_fingerprint(cfg),'Configuration fingerprint differs')
    need(state['data_fingerprint']==result['data_fingerprint']==saved['fingerprint'],'Data fingerprints differ')
    need(state['stages']==result['stages']and len(state['stages'])==10,'Checkpoint/result stage sequence differs')
    need(result['metrics']['completed_tasks']==10,'Need all ten development stages')
    derived=summarize(state['stages'])
    for key in ('final_accuracy','aia','forgetting_previous','forgetting_including_final'):
        need(abs(derived[key]-result['metrics'][key])<1e-12,'Development metric closure differs')
    for i,s in enumerate(state['stages']):
        need(s['task']==i and s['seen_classes']==(i+1)*10 and s['prediction_space']=='all_seen_classes_without_task_id','Stage vocabulary differs')
        need(len(s['task_counts'])==len(s['task_accuracy'])==10 and s['evaluation_images']==(i+1)*500,'Stage holdout counts differ')
        need(all(s['task_counts'][j]==500 and s['task_accuracy'][j]is not None for j in range(i+1)),'Seen task evaluation differs')
        need(all(s['task_counts'][j]==0 and s['task_accuracy'][j]is None for j in range(i+1,10)),'Future task leakage')
        need(abs(sum(s['task_accuracy'][:i+1])/(i+1)-s['accuracy'])<1e-12,'Sample-weighted accuracy differs')
    learner=state['learner'];core=learner['core'];engine=core['engine'];memory=core['memory'];stats=engine['statistics']
    need(learner['cur_task']==9 and learner['known_classes']==learner['total_classes']==100,'Not full development endpoint')
    need(engine['completed_tasks']==10 and engine['task_sizes']==[4500]*10 and stats['count']==45000,'Historical arrivals differ')
    need(stats['classes']==list(range(100))and stats['G'].shape==(769,769)and stats['C'].shape==(769,100),'Moment shapes/vocabulary differ')
    need(engine['W0'].shape==engine['V'].shape==(769,100),'Wrong deployed readout widths')
    need(engine['architecture']==dict(positions=(7,9),count=4,rank=8,scale=.1,width=768),'Fixed library architecture differs')
    need((engine['options']['outer_updates'],tuple(engine['options']['schedule']))==expected and engine['options']['episode_replay_batch_size']==64,'Stored episode options differ')
    records=memory['records'];ids={int(r.sample_id)for r in records}
    need(memory['seen']==45000 and memory['classes']==list(range(100))and not memory['fixed']and memory['memory_size']==2000,'Replay history/policy differs')
    need(len(records)==len(ids)==2000 and all(sum(r.label==c for r in records)==20 for c in range(100)),'Final resident IDs/quotas differ')
    need(set(engine['memory_base'])==ids and all(z.shape==(769,)and float(z[-1])==1 for z in engine['memory_base'].values()),'Resident anchor cache differs')
    diagnostics=learner['task_diagnostics'];need(len(diagnostics)==10 and diagnostics==result['task_diagnostics'],'Task diagnostics differ')
    need(all(d['backbone_changed']and d['modules_changed']and d['replay_images']==(i+1)*10*(2000//((i+1)*10))for i,d in enumerate(diagnostics)),'Learning/replay records differ')
    need(any(torch.count_nonzero(v)>0 for k,v in engine['encoder'].items()if k.startswith('libraries.')and k.endswith('.U')),'Libraries remain at zero U')
    need(engine['updates']==engine['module_updates']+engine['backbone_updates'],'Update counters differ')
    need(set(engine['optimizers'])=={'modules','backbone'},'Missing persistent AdamW family');optimizers={}
    for name,opt in engine['optimizers'].items():
        params=[x for g in opt['param_groups']for x in g['params']]
        need(len(params)==len(set(params))and set(params)==set(opt['state']),'Incomplete AdamW states')
        steps=[]
        for v in opt['state'].values():
            need({'step','exp_avg','exp_avg_sq'}<=set(v)and v['exp_avg'].shape==v['exp_avg_sq'].shape,'Malformed AdamW state')
            step=float(v['step']);need(step>0,'Invalid AdamW step');steps.append(step)
        optimizers[name]=dict(parameters=len(params),min_step=min(steps),max_step=max(steps))
    for key in ('rng','kind_rng'):torch.Generator(device='cpu').set_state(engine[key])
    generator=np.random.default_rng();generator.bit_generator.state=memory['rng']
    cuda_payloads=native.global_rng_guard(state,np,torch)
    from cl27.data import ProtocolDataManager
    need(Path(sys.modules[ProtocolDataManager.__module__].__file__).resolve().is_relative_to(repo),'DataManager source differs')
    data_args=copy.deepcopy(cfg);data_args['download']=False
    dm=ProtocolDataManager('cifar224',True,1993,10,10,data_args)
    need(dm.manifest==saved and dm.nb_classes==100 and dm.nb_tasks==10,'Actual development manifest differs')
    fit=np.asarray(dm._train_ids);holdout=np.asarray(dm._test_ids)
    need(len(fit)==45000 and len(holdout)==5000 and not set(fit)&set(holdout),'Development IDs overlap or counts differ')
    need(np.array_equal(np.sort(np.r_[fit,holdout]),np.arange(50000)),'Development partition does not cover original train IDs')
    counts=np.bincount(dm._train_targets,minlength=100)
    need(np.array_equal(counts,np.full(100,450))and np.array_equal(stats['C'][-1].numpy(),counts),'Class count row differs')
    row={int(uid):i for i,uid in enumerate(fit)}
    for r in records:
        need(int(r.sample_id)in row,'Resident is not a fitting member');i=row[int(r.sample_id)]
        need(r.label==int(dm._train_targets[i])and isinstance(r.image,np.ndarray)and r.image.dtype==np.uint8 and np.array_equal(r.image,dm._train_data[i]),'Resident label/RGB differs')
    restored=native.native_restore(cfg,state,dm,npz,comparator,np,torch)
    need(native.sha(backup)==digest and native.sha(npz)==native.WEIGHT_SHA,'Read-only source bytes changed')
    need(native.sha(auditor)==auditor_sha and native.sha(loader)==loader_sha,'Audit utility source changed during read')
    print(json.dumps(dict(status='selected_development_audit_checks_passed_not_formal_result',audit_type='selected_development_audit',formal_result_audit=False,final_audit_complete=False,
        selected_trial=a.selected_trial,phase='development',official_test_read=False,checkpoint_sha256=digest,
        wrapper_sha256=native.sha(Path(__file__)),native_auditor_sha256=auditor_sha,state_loader_sha256=loader_sha,executed_source_root=str(repo),
        metadata_sha256={n:native.sha(meta/(n+'.json'))for n in values},training_rows=45000,holdout_rows=5000,classes=100,final_replay_images=2000,
        finite_scan=finite,optimizers=optimizers,cuda_rng_payload_lengths=cuda_payloads,native_restore=restored,development_metrics={k:result['metrics'][k]for k in derived if k!='accuracy_matrix'},
        cpu_audit_wall_seconds=time.perf_counter()-began,limits=['Caller binds terminal selected process and root-reviewed selection to this finalized backup SHA.','No CUDA continuation,training,accuracy evaluation or formal-test read; source is the development train holdout.','This is not a formal pass; fresh formal refit must use all50000 training rows.']),indent=2))


if __name__=='__main__':main()
