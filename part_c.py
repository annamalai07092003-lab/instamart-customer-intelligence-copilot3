"""Part C: privacy-safe ticket triage with retries, output validation and token budget."""
from pathlib import Path
import re,json,random,time
import pandas as pd
import mock_llm
ROOT=Path(__file__).parent; RAW=ROOT/'data'/'raw'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
ALLOWED={'billing','delivery','technical','fraud'}; MAX_INPUT_TOKENS=400; BASE_DELAY=.05
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'); PHONE=re.compile(r'\b[6-9]\d{9}\b'); ORDER=re.compile(r'ORD-\d{6}')
ZERO_PROMPT='CLASSIFY each customer support ticket. Allowed categories: billing, delivery, technical, fraud. Return JSON only in exactly this shape: {"category": "...", "reason": "..."}. Do not include prose outside JSON.'
FEW_PROMPT='''CLASSIFY each customer support ticket. Allowed categories: billing, delivery, technical, fraud. Return JSON only in exactly this shape: {"category": "...", "reason": "..."}. Do not include prose outside JSON.
EXAMPLES:
Ticket: I was charged twice and need a refund.
Label: {"category":"billing","reason":"duplicate charge"}
Ticket: There is an unauthorized transaction and my account was hacked.
Label: {"category":"fraud","reason":"unauthorized transaction"}
Ticket: The app crashes when I open it.
Label: {"category":"technical","reason":"app crash"}'''
def redact(text):
    counts={}; text,n=EMAIL.subn('[EMAIL]',text); counts['emails']=n
    text,n=PHONE.subn('[PHONE]',text); counts['phones']=n
    text,n=ORDER.subn('[ORDER_ID]',text); counts['order_ids']=n
    return text,counts
def assert_redacted(text):
    assert not EMAIL.search(text) and not PHONE.search(text) and not ORDER.search(text), 'PII remains after redaction'
def retry_chat(messages,ticket_id):
    for retry_no in range(4):
        try: return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_no>=3: raise
            wait=BASE_DELAY*(2**retry_no)*random.uniform(.9,1.15)
            print(f'RETRY ticket_id={ticket_id} retry={retry_no} wait={wait:.4f}s')
            time.sleep(wait)
def classify_reply(content):
    try:
        obj=json.loads(content or '')
        if isinstance(obj,dict) and obj.get('category') in ALLOWED:
            return obj['category'], 'classified_json'
        return '', 'flagged_invalid_output'
    except (json.JSONDecodeError,TypeError):
        words=re.findall(r'[a-z]+',(content or '').lower()); found=ALLOWED.intersection(words)
        if len(found)==1: return next(iter(found)),'classified_keyword'
        return '', 'flagged_invalid_output'
def run_pipeline(prompt,token_budget=None,segments=True,save_name=None):
    tickets=pd.read_csv(RAW/'tickets.csv',keep_default_na=False); random.seed(42); mock_llm.reset()
    results=[]; token_total=0; pii_counts={'emails':0,'phones':0,'order_ids':0}; eligible_calls=0
    seg_path=OUT/'customer_segments.csv'; seg={}
    if segments and seg_path.exists():
        sf=pd.read_csv(seg_path); seg={r.customer_id:r for r in sf.itertuples(index=False)}
    for row in tickets.itertuples(index=False):
        raw=row.text; category=''; status=''; used=0; redacted,counts=redact(raw)
        for k,v in counts.items(): pii_counts[k]+=v
        assert_redacted(redacted)
        if not raw.strip(): status='skipped_empty'
        elif mock_llm.est_tokens(raw)>MAX_INPUT_TOKENS: status='flagged_long'
        elif token_budget is not None and token_total>=token_budget: status='budget_exhausted'
        else:
            # The user message is exactly the redacted ticket text, with no metadata added.
            messages=[{'role':'system','content':prompt},{'role':'user','content':redacted}]
            assert messages[-1]['content']==redacted
            response=retry_chat(messages,row.ticket_id); eligible_calls+=1
            used=response.usage.total_tokens; token_total+=used
            category,status=classify_reply(response.choices[0].message.content)
        segment=seg.get(row.customer_id); seg_name=getattr(segment,'segment_name','') if segment else ''; premium=bool(getattr(segment,'is_premium',False)) if segment else False
        if status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}: route='human_review'; priority='NORMAL'
        elif category=='fraud': route='security_team'; priority='HIGH'
        elif premium: route='premium_desk'; priority='HIGH'
        else: route='support_queue'; priority='NORMAL'
        results.append({'ticket_id':row.ticket_id,'customer_id':row.customer_id,'status':status,'category':category,'segment_name':seg_name,'priority':priority,'route':route,'tokens_used':used,'true_category':row.true_category,'redacted_text':redacted})
    df=pd.DataFrame(results)
    if save_name: df.drop(columns=['true_category','redacted_text']).to_csv(OUT/save_name,index=False)
    classified=df[df.status.isin(['classified_json','classified_keyword'])]
    accuracy=float((classified.category==classified.true_category).mean()) if len(classified) else 0
    summary={'tickets_total':len(df),'classified':len(classified),'status_counts':df.status.value_counts().to_dict(),'accuracy_classified':accuracy,'tokens_total':token_total,'tokens_per_classified':token_total/max(1,len(classified)),'eligible_model_calls':eligible_calls,'pii_substitutions':pii_counts,'pii_total':sum(pii_counts.values()),'pii_remaining_after_redaction':0,'budget_exhausted':int(df.status.eq('budget_exhausted').sum())}
    return df,summary

def run():
    z,zsum=run_pipeline(ZERO_PROMPT,None,False)
    f,fsum=run_pipeline(FEW_PROMPT,None,True,'triage_results.csv')
    # confusion matrix for classified zero-shot tickets
    zc=z[z.status.isin(['classified_json','classified_keyword'])]
    pd.crosstab(zc.true_category,zc.category,dropna=False).to_csv(OUT/'zero_shot_confusion.csv')
    bz,bsum=run_pipeline(ZERO_PROMPT,6000,False,'triage_budget_demo.csv')
    # Ensure a budget demonstration exhausts; if this prompt's exact token usage does not, try 5000 as brief permits.
    budget_used=6000
    if bsum['budget_exhausted']==0:
        bz,bsum=run_pipeline(ZERO_PROMPT,5000,False,'triage_budget_demo.csv'); budget_used=5000
    report={'zero_shot':zsum,'few_shot':fsum,'budget_demo':bsum,'budget_used':budget_used}
    (OUT/'part_c_report.json').write_text(json.dumps(report,indent=2))
    print('Part C complete:',json.dumps({'zero_shot':zsum,'few_shot':fsum,'budget_used':budget_used,'budget_exhausted':bsum['budget_exhausted']},indent=2))
if __name__=='__main__': run()
