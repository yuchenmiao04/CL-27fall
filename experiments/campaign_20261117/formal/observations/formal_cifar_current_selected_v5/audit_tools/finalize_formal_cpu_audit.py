"""Finalize one audited formal cell; standard library only, never trains/copies models.
--check-only validates existing evidence without writes. A real invocation refuses
existing receipt/archive. --audit-json defaults to the audit job's JSON stdout log.
"""
import argparse, datetime, hashlib, importlib.util, io, json, os, sys, tarfile, tempfile
from pathlib import Path
ALLOWED = {
 'formal_cifar_er_fixed_v1':('er','cifar224','runtime_fixed_er_v5','root'),
 'formal_imagenetr_er_fixed_v1':('er','imagenetr','runtime_fixed_er_v5','root'),
 'formal_cifar_transport_selected_v2':('history_transport','cifar224','runtime_tuning_v1','data'),
 'formal_imagenetr_transport_selected_v3':('history_transport','imagenetr','runtime_tuning_v1','data'),
 'formal_cifar_plastic_shared_selected_v5':('plastic_shared_basis','cifar224','runtime_tuning_v1','data'),
 'formal_imagenetr_plastic_shared_selected_v5':('plastic_shared_basis','imagenetr','runtime_tuning_v1','data'),
 'formal_cifar_current_selected_v5':('current_adapted','cifar224','runtime_bounded_v4','data'),
 'formal_imagenetr_current_selected_v5':('current_adapted','imagenetr','runtime_bounded_v4','data')}
def need(ok, message):
 if not ok: raise ValueError(message)
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4<<20),b''): h.update(x)
 return h.hexdigest()
def terminal(receipt):
 pid,ticks=receipt['pid'],str(receipt['proc_start_ticks']); need(type(pid) is int and pid>0 and ticks.isdigit(),'Bad original PID identity')
 try: text=(Path('/proc')/str(pid)/'stat').read_text(); fields=text[text.rfind(')')+2:].split()
 except (FileNotFoundError,ProcessLookupError): return dict(pid=pid,expected_ticks=ticks,state='absent')
 need(fields[19]!=ticks or fields[0] in {'Z','X','x'},'Original process is still live')
 return dict(pid=pid,expected_ticks=ticks,observed_ticks=fields[19],state=fields[0])
def cache_proof(base, job, method, dataset, result):
 if method=='current_adapted':
  need(not result.get('execution',{}).get('ce_training_reuse'),'Current must not use CE trajectory replay'); return None,{}
 prefix='cifar' if dataset=='cifar224' else 'imagenetr'; donor='formal_'+prefix+'_er_fixed_v1'
 er=base/'formal/runs'/donor; invpath=base/'formal/cohorts'/(donor+'.cache_inventory.json'); inv=read(invpath); dr=read(er/'results.json')
 dc=read(er/'config.json'); need(dc['model_name']=='er' and dc['dataset']==dataset and dc.get('execution_role')=='fixed_baseline' and dr['status']=='complete' and dr['phase']=='final','Wrong formal ER donor')
 need(inv['status']=='closed_formal_er_cache_inventory' and inv['donor']==donor and inv['source_results_sha256']==sha(er/'results.json'),'Unbound ER inventory')
 if method!='er':
  prior=read(base/'formal/completion_receipts'/(donor+'.json')); need(prior['cache_inventory_sha256']==sha(invpath) and prior['results_sha256']==sha(er/'results.json'),'Donor inventory is not sealed by its audited completion')
 reuse=result['execution']['ce_training_reuse']; native=dr['execution']['ce_training_reuse']; directory=base/'formal/ce_trajectories'/(prefix+'_fixed_er_v1')
 need(reuse['mode']==('record' if method=='er' else 'replay') and reuse['execution_role']==('fixed_baseline' if method=='er' else 'candidate_method'),'Wrong reuse mode/role')
 need(native['mode']=='record' and native['execution_role']=='fixed_baseline' and reuse['trajectory_directory']==native['trajectory_directory']==str(directory),'Wrong formal cache directory')
 rows,original,sealed=reuse['stages'],native['stages'],inv['cache_files']; need(all(len(x)==10 and [r['task'] for r in x]==list(range(10)) for x in (rows,original,sealed)),'Need exactly ten stages')
 for i,(row,old,entry) in enumerate(zip(rows,original,sealed)):
  need(row['reused'] is (method!='er') and old['reused'] is False,'Wrong stage reuse flag')
  for key in ('task','path','request_fingerprint','body_after_sha256','head_after_sha256','shared_native_ce_seconds'): need(row[key]==old[key]==entry[key],'Reader/donor/inventory mismatch: '+key)
  path=directory/f'stage_{i:03d}.pt'; need(row['path']==str(path) and path.is_file() and path.stat().st_size==entry['file_bytes'] and sha(path)==entry['file_sha256'],'Actual sealed cache bytes differ')
 need(sum(r['shared_native_ce_seconds'] for r in rows)==inv['shared_native_ce_seconds'],'Shared CE cost sum differs')
 extra={f'formal/cohorts/{invpath.name}':invpath}
 if method!='er':
  extra.update({f'formal/runs/{donor}/config.json':er/'config.json',f'formal/runs/{donor}/results.json':er/'results.json',f'formal/completion_receipts/{donor}.json':base/'formal/completion_receipts'/(donor+'.json')})
 return dict(donor=donor,stages=10,reader_reused=method!='er',actual_payload_hashes_verified=True,shared_native_ce_seconds=inv['shared_native_ce_seconds'],inventory_sha256=sha(invpath)),extra
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--job',required=True,choices=ALLOWED); p.add_argument('--audit-job',required=True)
 p.add_argument('--audit-json',type=Path); p.add_argument('--validator',type=Path); p.add_argument('--auditor-source',type=Path); p.add_argument('--output',type=Path); p.add_argument('--check-only',action='store_true'); a=p.parse_args()
 need(Path(a.audit_job).name==a.audit_job and a.audit_job.startswith('audit_'+a.job+'_'),'Audit job must be the explicit matching cell audit name')
 root=Path('remote_runtime'); base=root/'campaign_20261117'; method,dataset,runtime,volume=ALLOWED[a.job]; repo=root/runtime
 run=base/'formal/runs'/a.job; trainpath=base/'formal/receipts'/(a.job+'.json'); auditpath=base/'verification/receipts'/(a.audit_job+'.json')
 backup=(Path('/root/cl27-formal-backups') if volume=='root' else root/'formal_backups')/'campaign_20261117'/a.job
 cfg,result,data=(read(run/(n+'.json')) for n in ('config','results','data_manifest')); train,ar,bv=read(trainpath),read(auditpath),read(backup/'verification.json')
 need(train['job']==a.job and ar['job']==a.audit_job and train['cwd']==ar['cwd']==str(repo),'Job/runtime receipt identity differs')
 need(ar['metadata']['original_formal_job']==a.job and ar['metadata']['execution_role']=='native_cpu_final_checkpoint_audit','Audit is for another cell')
 ended={a.job:terminal(train),a.audit_job:terminal(ar)}
 auditjson=a.audit_json or base/'verification/logs'/(a.audit_job+'.log'); audit=read(auditjson)
 need(audit['status']=='native_cpu_checkpoint_audit_checks_passed_not_campaign_certificate' and audit['executed_model_source_root']==str(repo),'Native CPU audit is not PASS for this runtime')
 need(audit['auditor_sha256']==ar['metadata']['auditor_sha256'] and audit['native_restore']['native_cpu_restore_executed'] is True and audit['native_restore']['raw_checkpoint_numerically_restored'] is True,'Native restore proof missing')
 need(audit['native_restore']['extra_training_steps']==0 and audit['native_restore']['accuracy_evaluation_performed'] is False and audit['native_restore']['cuda_rng_applied'] is False and audit['native_restore']['output_is_finite'] is True,'Audit scope differs')
 need(cfg['model_name']==result['method']==method and cfg['dataset']==dataset and cfg['seed']==1993 and cfg['phase']==result['phase']=='final' and result['status']=='complete','Not the selected completed formal cell')
 need(not cfg.get('validation_only',False) and not cfg.get('smoke',False) and all(cfg.get(k) is None for k in ('max_tasks','resume','validation_max_per_class','joint_max_classes')),'Partial/development/resumed configuration')
 need(method!='er' or cfg.get('execution_role')=='fixed_baseline','Internal CE driver is not formal ER')
 n,e,c=(50000,10000,100) if dataset=='cifar224' else (24000,6000,200)
 need(data['train_count']==n and data['evaluation_count']==e and data['phase']=='final' and data['official_test_read'] is True and data['fingerprint']==result['data_fingerprint'],'Wrong full formal data scope')
 need(result['stages'][-1]['evaluation_images']==e and result['stages'][-1]['seen_classes']==c,'Final evaluation count/output vocabulary differs')
 need(audit['facts']['method']==method and audit['facts']['dataset']==dataset and audit['facts']['history_rows']==n and audit['facts']['classes']==c and audit['facts']['final_replay_rows']==20*c,'CPU audit cell/counts differ')
 need(audit['data_proof']['manifest_rebuilt_from_actual_source_bytes'] is True and audit['native_restore']['output_shape']==[1,c],'Missing actual-source/all-class audit')
 for name in ('config','results','data_manifest'):
  need(sha(run/(name+'.json'))==bv['source_metadata_sha256'][name]==audit['metadata_sha256'][name]==sha(backup/(name+'.json')),'Metadata changed after backup/CPU audit: '+name)
 need(bv['status']=='verified_server_local_backup' and bv['job']==a.job and bv['payload_left_server'] is False and bv['source_receipt_sha256']==sha(trainpath)==sha(backup/'receipt.json'),'Backup receipt provenance differs')
 need(bv['source_checkpoint']==str(run/'checkpoint.pt') and bv['backup_checkpoint']==audit['checkpoint_backup']==str(backup/'checkpoint.pt'),'Backup path differs from approved volume')
 for path in (run/'checkpoint.pt',backup/'checkpoint.pt'):
  need(path.is_file() and path.stat().st_size==bv['checkpoint_bytes'] and sha(path)==bv['checkpoint_sha256']==bv['backup_actual_sha256']==audit['checkpoint_sha256']==ar['metadata']['checkpoint_sha256'],'Actual checkpoint SHA differs')
 validator=a.validator or repo/'scripts/summarize_campaign.py'; sys.dont_write_bytecode=True
 spec=importlib.util.spec_from_file_location('formal_metric_closure_validator',validator); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); metrics=module.validate_complete(cfg,result)
 cache,extra=cache_proof(base,a.job,method,dataset,result)
 paths={f'formal/runs/{a.job}/{name}.json':run/(name+'.json') for name in ('config','results','data_manifest')}
 paths.update(extra); paths.update({f'formal/receipts/{trainpath.name}':trainpath,f'verification/receipts/{auditpath.name}':auditpath,f'verification/logs/{a.audit_job}.log':auditjson,f'server_backup/{a.job}/verification.json':backup/'verification.json'})
 for name,expected in audit['runtime_source_files_sha256'].items():
  rel=Path(name); need(not rel.is_absolute() and '..' not in rel.parts and rel.suffix=='.py','Unsafe runtime source name'); source=repo/rel; need(sha(source)==expected,'Actual runtime source changed: '+name); paths['runtime_source/'+name]=source
 auditor=a.auditor_source or Path(ar['command'][1]); loader=Path(audit['state_loader'])
 need(sha(auditor)==audit['auditor_sha256'] and sha(loader)==audit['state_loader_sha256'],'Observer sources differ from successful audit')
 paths.update({'audit_tools/native_cpu_auditor.py':auditor,'audit_tools/state_loader.py':loader,'audit_tools/metric_validator.py':validator,'audit_tools/finalize_formal_cpu_audit.py':Path(__file__)})
 for path in (base/'formal/logs'/(a.job+'.log'),run/'training.log'):
  if path.is_file(): paths[str(path.relative_to(base))]=path
 receipt=base/'formal/completion_receipts'/(a.job+'.json'); output=a.output or base/'formal/audited_metadata'/(a.job+'.tar.gz')
 proof=dict(status='audited_complete_formal_run',job=a.job,dataset=dataset,method=method,seed=1993,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),metrics=metrics,full_training_rows=n,final_evaluation_samples=e,final_replay_rows=20*c,results_sha256=sha(run/'results.json'),checkpoint_sha256=bv['checkpoint_sha256'],checkpoint_bytes=bv['checkpoint_bytes'],backup_location=bv['backup_checkpoint'],backup_payload_left_server=False,original_processes_terminal=ended,cpu_audit_job=a.audit_job,cpu_audit_json_sha256=sha(auditjson),validator_source_sha256=sha(validator),cache_reuse_verification=cache,scope='This one formal cell only; not a twelve-cell campaign certificate; no extra training, GPU continuation or accuracy rerun.')
 proof.update(source_process_terminal=ended[a.job],audit_process_terminal=ended[a.audit_job],final_correct=round(metrics['final_accuracy']*e),cpu_audit=dict(job=a.audit_job,log_sha256=sha(auditjson),auditor_sha256=audit['auditor_sha256'],native_cpu_restore=True,compared_state_nodes=audit['native_restore']['compared_state_nodes'],actual_source_data_and_replay_verified=True,no_extra_training_or_accuracy_evaluation=True,cuda_rng_payloads_retained_not_applied=True))
 if cache: proof['cache_inventory_sha256']=cache['inventory_sha256']
 if a.check_only: print(json.dumps(dict(status='check_only_pass',writes=False,job=a.job,audit_job=a.audit_job,existing_receipt=receipt.exists(),evidence=proof))); return
 need(not receipt.exists() and not output.exists(),'Refusing to overwrite existing completion receipt/archive')
 receipt.parent.mkdir(parents=True,exist_ok=True); output.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='formal-metadata-',dir=output.parent) as temp:
  payload=(json.dumps(proof,indent=2,allow_nan=False)+'\n').encode(); rp=Path(temp)/'receipt.json'; rp.write_bytes(payload); tarpath=Path(temp)/'metadata.tar.gz'; manifest={}
  with tarfile.open(tarpath,'w:gz') as archive:
   def add(name,content):
    info=tarfile.TarInfo(name); info.size=len(content); info.mode=0o644; archive.addfile(info,io.BytesIO(content)); manifest[name]=dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
   for name,path in sorted(paths.items()):
    need(path.suffix in {'.json','.log','.py'},'Only explicit JSON/log/source files may be exported'); add(name,path.read_bytes())
   add(str(receipt.relative_to(base)),payload); add('EXPORT_MANIFEST.json',(json.dumps(dict(job=a.job,weights_included=False,files=manifest),indent=2)+'\n').encode())
  os.link(rp,receipt)
  try: os.link(tarpath,output)
  except BaseException: receipt.unlink(); raise
 print(json.dumps(dict(status='single_formal_cell_finalized',job=a.job,receipt=str(receipt),metadata_archive=str(output),archive_bytes=output.stat().st_size,archive_sha256=sha(output),weights_included=False)))
if __name__=='__main__': main()
