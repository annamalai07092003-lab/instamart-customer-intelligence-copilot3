"""Part D: policy retrieval, grounded QA and resumable refund agent."""
from pathlib import Path
import json,re,time,sqlite3,shutil
from datetime import datetime,timezone
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
ROOT=Path(__file__).parent; POL=ROOT/'data'/'policies'; RAW=ROOT/'data'/'raw'; OUT=ROOT/'outputs'; STATE=ROOT/'state'; OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
class SimulatedCrash(RuntimeError): pass

def chunk_documents(overlap=10):
    chunks=[]
    for path in sorted(POL.glob('*.txt')):
        words=path.read_text(encoding='utf-8').split(); start=0
        while start<len(words):
            end=min(start+40,len(words)); chunks.append({'source':path.name,'text':' '.join(words[start:end])})
            if end>=len(words): break
            start=end-overlap
    return chunks
class PolicyRAG:
    def __init__(self,overlap=10):
        self.chunks=chunk_documents(overlap); self.vectorizer=TfidfVectorizer(); self.matrix=self.vectorizer.fit_transform([c['text'] for c in self.chunks])
    def retrieve(self,query,k=3):
        q=self.vectorizer.transform([query]); scores=cosine_similarity(q,self.matrix)[0]; order=scores.argsort()[::-1][:k]
        return [{'source':self.chunks[i]['source'],'score':float(scores[i]),'text':self.chunks[i]['text']} for i in order]
    def answer(self,question):
        hits=self.retrieve(question,3)
        context='\n'.join(f"[{h['source']}]\n{h['text']}" for h in hits)
        user=f'CONTEXT:\n{context}\nQUESTION:\n{question}'
        resp=mock_llm.chat([{'role':'system','content':'POLICY ASSISTANT: Answer only from the supplied context. If the answer is not present in the context, say exactly: I could not find that information in the policy documents.'},{'role':'user','content':user}])
        return resp.choices[0].message.content,[h['source'] for h in hits]

def policy_eval(rag,label):
    df=pd.read_csv(RAW/'policy_eval.csv',keep_default_na=False); rows=[]
    for r in df.itertuples(index=False):
        hits=rag.retrieve(r.question,3); ans,sources=rag.answer(r.question)
        rows.append({'question':r.question,'expected_doc':r.expected_doc,'retrieved_sources':'|'.join(sources),'expected_doc_in_top3':r.expected_doc=='NONE' or r.expected_doc in sources,'expected_fact_found':bool(r.expected_fact and r.expected_fact in ans),'fallback':ans.startswith('I could not find'),'answer':ans})
    pd.DataFrame(rows).to_csv(OUT/f'policy_eval_{label}.csv',index=False)
    answerable=[x for x in rows if x['expected_doc']!='NONE']; unanswerable=[x for x in rows if x['expected_doc']=='NONE']
    return {'answerable_doc_retrieval':f"{sum(x['expected_doc_in_top3'] for x in answerable)}/{len(answerable)}",'answerable_expected_fact':f"{sum(x['expected_fact_found'] for x in answerable)}/{len(answerable)}",'unanswerable_fallback':f"{sum(x['fallback'] for x in unanswerable)}/{len(unanswerable)}"}

TOOLS_SPEC=[
 {'name':'get_order','description':'Look up a single order in the SQLite database.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id']}},
 {'name':'search_policy','description':'Retrieve the most relevant policy chunks for a query.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},
 {'name':'calculate_refund','description':'Calculate the refund amount using late-delivery rules.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':'number'}},'required':['order_amount','delivery_minutes']}},
 {'name':'issue_refund','description':'Append an automatically approved refund action to the ledger.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},
 {'name':'escalate_to_human','description':'Append a refund escalation to the human support ledger.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}}]
_ACTIVE_RAG=None

def get_order(order_id):
    with sqlite3.connect(ROOT/'data'/'instamart.db') as con:
        df=pd.read_sql_query('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',con,params=(order_id,))
    if df.empty: return {'error':'order not found'}
    r=df.iloc[0].to_dict(); return {'order_id':r['order_id'],'status':r['status'],'order_amount':float(r['order_amount']),'delivery_minutes':None if pd.isna(r['delivery_minutes']) else float(r['delivery_minutes'])}
def search_policy(query): return {'results':_ACTIVE_RAG.retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes): return {'refund_inr':round(float(order_amount),2) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0}
def _ledger(tool_name,order_id,refund_inr):
    row={'timestamp':datetime.now(timezone.utc).isoformat(),'tool_name':tool_name,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
    with (OUT/'refund_ledger.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
    return {'refund_inr':round(float(refund_inr),2)}
def issue_refund(order_id,refund_inr): return _ledger('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return _ledger('escalate_to_human',order_id,refund_inr)
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}

def _status(answer): return (answer.split(':',1)[0] if answer else 'FALLBACK').strip()
def _fallback(status,summary): return f'{status}: Human handoff required. {summary}'
def _checkpoint(run_id,scenario_id,messages,tokens_used,steps,tools_executed,step_number):
    data={'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens_used,'steps':steps,'tools_executed':tools_executed,'step_number':step_number}
    (STATE/f'{run_id}.json').write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')
    return data

def _agent_loop(run_id,scenario_id,messages,tokens_used,steps,tools_executed,max_iterations,token_limit,time_limit_seconds,start_time,crash_after_tool_count=None):
    llm_calls=0
    while llm_calls<max_iterations:
        if time.monotonic()-start_time>=time_limit_seconds:
            answer=_fallback('FALLBACK_TIMEOUT','Execution time limit reached.'); return {'status':'FALLBACK_TIMEOUT','final_answer':answer,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed}
        response=mock_llm.chat(messages,tools=TOOLS_SPEC); llm_calls+=1; tokens_used+=int(response.usage.total_tokens)
        if tokens_used>token_limit:
            answer=_fallback('FALLBACK_TOKEN_LIMIT','Token budget exceeded before executing the requested tool.'); return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':answer,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed}
        msg=response.choices[0].message; calls=getattr(msg,'tool_calls',[]) or []
        if not calls:
            answer=msg.content or _fallback('FALLBACK','The assistant returned no final answer.')
            return {'status':_status(answer),'final_answer':answer,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed}
        # mock returns one tool call per response; handle sequentially.
        serialized=[]
        for call in calls:
            name=call.name; args=call.arguments
            if isinstance(args,str): args=json.loads(args)
            serialized.append({'id':call.id,'name':name,'arguments':args})
        messages.append({'role':'assistant','content':None,'tool_calls':serialized})
        for call in serialized:
            name=call['name']; args=call['arguments']
            if name not in TOOL_MAP:
                observation={'error':f'unknown tool {name}'}
            else:
                observation=TOOL_MAP[name](**args)
            step={'tool_name':name,'arguments':args,'observation':observation}; steps.append(step); tools_executed.append(name)
            messages.append({'role':'tool','tool_call_id':call['id'],'name':name,'content':json.dumps(observation,default=str)})
            _checkpoint(run_id,scenario_id,messages,tokens_used,steps,tools_executed,len(steps))
            if crash_after_tool_count is not None and len(tools_executed)==crash_after_tool_count:
                raise SimulatedCrash(f'Simulated crash after tool execution {crash_after_tool_count}; checkpoint saved.')
    answer=_fallback('FALLBACK_MAX_ITERATIONS','Maximum model-call iterations reached. Please review the case.')
    return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':answer,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_executed':tools_executed}

def run_agent(user_message,scenario_id, max_iterations=6, token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tool_count=None):
    run_id=run_id or f'{scenario_id}_{int(time.time()*1000)}'; messages=[{'role':'system','content':'AGENT: Use tools to handle refund requests according to policy. Only completed orders may be refunded; >45 delivery minutes qualifies; amounts above INR 1000 must be escalated.'},{'role':'user','content':user_message}]
    return _agent_loop(run_id,scenario_id,messages,0,[],[],max_iterations,token_limit,time_limit_seconds,time.monotonic(),crash_after_tool_count)

def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    path=STATE/f'{run_id}.json'; data=json.loads(path.read_text(encoding='utf-8'))
    return _agent_loop(run_id,data['scenario_id'],data['messages'],data['tokens_used'],data['steps'],data['tools_executed'],max_iterations,token_limit,time_limit_seconds,time.monotonic(),None)

def run():
    global _ACTIVE_RAG
    _ACTIVE_RAG=PolicyRAG(10)
    chunk_counts=pd.Series([c['source'] for c in _ACTIVE_RAG.chunks]).value_counts().to_dict()
    retrieval=policy_eval(_ACTIVE_RAG,'overlap10')
    rag0=PolicyRAG(0); zero=policy_eval(rag0,'overlap0')
    (OUT/'part_d_rag_report.json').write_text(json.dumps({'chunk_count':len(_ACTIVE_RAG.chunks),'chunk_counts':chunk_counts,'overlap10':retrieval,'overlap0':zero},indent=2))
    scenarios=pd.read_csv(RAW/'agent_scenarios.csv',keep_default_na=False)
    (OUT/'refund_ledger.jsonl').write_text('',encoding='utf-8'); (OUT/'agent_runs.jsonl').write_text('',encoding='utf-8')
    scenario_rows=[]
    for r in scenarios.itertuples(index=False):
        result=run_agent(r.message,r.scenario_id)
        record={'scenario_id':r.scenario_id,'status':result['status'],'final_answer':result['final_answer'],'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used'],'steps':result['steps']}
        with (OUT/'agent_runs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        amount=next((s['observation'].get('refund_inr') for s in result['steps'] if s['tool_name']=='calculate_refund'),0)
        scenario_rows.append({'scenario_id':r.scenario_id,'status':result['status'],'refund_amount_inr':amount,'expected_outcome':r.expected_outcome,'expected_refund_inr':float(r.expected_refund_inr),'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used']})
    pd.DataFrame(scenario_rows).to_csv(OUT/'agent_scenario_results.csv',index=False)
    # Stop-safely demonstrations: S1 token limit, then zero-second timeout.
    s1=scenarios.iloc[0]; token_demo=run_agent(s1.message,'S1_TOKEN_DEMO',token_limit=200,run_id='demo_token_limit')
    timeout_demo=run_agent(s1.message,'S1_TIMEOUT_DEMO',time_limit_seconds=0,run_id='demo_timeout')
    # Crash after second tool observation is checkpointed, then resume; S1 executes exactly 4 tools overall.
    crash_id='S1_CRASH_RESUME'; crash_tools=[]
    try: run_agent(s1.message,'S1_CRASH_RESUME',run_id=crash_id,crash_after_tool_count=2)
    except SimulatedCrash: pass
    checkpoint=json.loads((STATE/f'{crash_id}.json').read_text()); crash_tools.extend(checkpoint['tools_executed'])
    resumed=resume_agent(crash_id); combined=resumed['tools_executed']
    crash_report={'status':resumed['status'],'final_answer':resumed['final_answer'],'tools_executed_across_crash_and_resume':combined,'llm_calls_after_resume':resumed['llm_calls'],'tokens_used_after_resume':resumed['tokens_used']}
    (OUT/'agent_safety_demos.json').write_text(json.dumps({'token_limit_demo':{k:token_demo[k] for k in ['status','final_answer','llm_calls','tokens_used']},'timeout_demo':{k:timeout_demo[k] for k in ['status','final_answer','llm_calls','tokens_used']},'crash_resume':crash_report},indent=2))
    print('Part D complete:',json.dumps({'chunks':len(_ACTIVE_RAG.chunks),'rag_overlap10':retrieval,'rag_overlap0':zero,'agent_statuses':[r['status'] for r in scenario_rows],'token_demo':token_demo['status'],'timeout_demo':timeout_demo['status'],'crash_resume':crash_report},indent=2))
if __name__=='__main__': run()
