"""CPU checks against exact pinned vLLM merge/build methods and native Jinja."""
import ast, copy, hashlib, json, pathlib, types, unittest
from typing import Any
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from request_adapter import adapt_request_body, InvalidRequestBody

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMAGE = ROOT/'patch-audit/image-vllm'
MODEL = 'qwen38-27b-nvidia-candidate'


def load_pinned_methods():
    sources=[IMAGE/'renderers/params.py', IMAGE/'entrypoints/openai/chat_completion/protocol.py']
    nodes=[]
    for source in sources:
        tree=ast.parse(source.read_text())
        wanted='merge_kwargs' if source.name=='params.py' else 'build_chat_params'
        nodes.append(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==wanted))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    env={'Any':Any,'ChatParams':lambda **kwargs:types.SimpleNamespace(**kwargs)}
    exec(compile(ast.fix_missing_locations(module),'<exact-pinned-vllm-methods>','exec'),env)
    return env['build_chat_params']


build_params=load_pinned_methods()
template_path=pathlib.Path('/home/frank/ai/models/qwen38-27b-nvidia-nvfp4/chat_template.jinja')
env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols'])
def fail(message):raise jinja2.TemplateError(message)
env.globals['raise_exception']=fail
env.filters['tojson']=lambda value,**kwargs:json.dumps(value,ensure_ascii=False,**kwargs)
template=env.from_string(template_path.read_text())


def effective(body):
    raw=json.loads(body)
    attrs=dict(add_generation_prompt=True,continue_final_message=False,documents=None,reasoning_effort=None,chat_template_kwargs=None,chat_template=None,media_io_kwargs=None,return_assistant_tokens_mask=False,tool_choice='none',tools=None,response_format=None)
    attrs.update({k:v for k,v in raw.items() if k in attrs})
    return build_params(types.SimpleNamespace(**attrs),None,'auto').chat_template_kwargs


def render(body):
    raw=json.loads(body)
    return template.render(messages=raw.get('messages',[dict(role='user',content='Reply READY.')]),**effective(body))


def encode(**kwargs):return json.dumps(dict(model=MODEL,**kwargs)).encode()


class AdapterTests(unittest.TestCase):
    def test_high_uses_native_xhigh(self):
        before=encode(reasoning_effort='high')
        with self.assertRaises(jinja2.TemplateError):render(before)
        after=adapt_request_body(before,candidate_model_id=MODEL)
        self.assertEqual(after.changed_field,'reasoning_effort')
        self.assertEqual(render(after.body),render(encode(reasoning_effort='xhigh')))

    def test_top_high_overrides_nested_low(self):
        before=encode(reasoning_effort='high',chat_template_kwargs=dict(reasoning_effort='low'))
        after=adapt_request_body(before,candidate_model_id=MODEL)
        self.assertEqual(effective(after.body)['reasoning_effort'],'xhigh')
        self.assertEqual(json.loads(after.body)['chat_template_kwargs']['reasoning_effort'],'low')
        render(after.body)

    def test_top_low_shadows_nested_high_without_mutation(self):
        before=encode(reasoning_effort='low',chat_template_kwargs=dict(reasoning_effort='high'))
        self.assertEqual(adapt_request_body(before,candidate_model_id=MODEL).body,before)
        self.assertEqual(effective(before)['reasoning_effort'],'low')
        render(before)

    def test_none_or_absent_top_uses_nested_high(self):
        for kwargs in [{},dict(reasoning_effort=None)]:
            before=encode(chat_template_kwargs=dict(reasoning_effort='high'),**kwargs)
            after=adapt_request_body(before,candidate_model_id=MODEL)
            self.assertEqual(after.changed_field,'chat_template_kwargs.reasoning_effort')
            self.assertEqual(effective(after.body)['reasoning_effort'],'xhigh')
            render(after.body)

    def test_supported_and_unknown_values_untouched(self):
        for effort in ['low','medium','xhigh','none','minimal','max',None]:
            before=encode(reasoning_effort=effort)
            self.assertEqual(adapt_request_body(before,candidate_model_id=MODEL).body,before)
        before=encode();self.assertEqual(adapt_request_body(before,candidate_model_id=MODEL).body,before)

    def test_invalid_top_auto_does_not_activate_nested_alias(self):
        before=encode(reasoning_effort='auto',chat_template_kwargs=dict(reasoning_effort='high'))
        self.assertEqual(adapt_request_body(before,candidate_model_id=MODEL).body,before)

    def test_disabled_thinking_preserved(self):
        before=encode(reasoning_effort='high',chat_template_kwargs=dict(enable_thinking=False))
        after=adapt_request_body(before,candidate_model_id=MODEL)
        self.assertEqual(effective(after.body)['enable_thinking'],False)
        self.assertEqual(render(before),render(after.body))

    def test_exact_other_bytes_stream_images_tools_and_number_spellings(self):
        before=(b'{ "model":"qwen38-27b-nvidia-candidate", "reasoning_effort" : "high",'
          b' "stream":true,"stream_options":{"include_usage":true},"temperature":1.00e-1,'
          b'"messages":[{"role":"developer","content":"Keep role unchanged"},'
          b'{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,high=="}}]}],'
          b'"tools":[{"type":"function","function":{"name":"high","parameters":{"type":"object"}}}] }')
        after=adapt_request_body(before,candidate_model_id=MODEL)
        expected=before.replace(b'"reasoning_effort" : "high"',b'"reasoning_effort" : "xhigh"')
        self.assertEqual(after.body,expected)

    def test_scope_other_model_unchanged(self):
        before=json.dumps(dict(model='qwen38-flash-next-reap320',reasoning_effort='high')).encode()
        self.assertEqual(adapt_request_body(before,candidate_model_id=MODEL).body,before)

    def test_unicode_and_escaped_high(self):
        before=b'{"model":"qwen38-27b-nvidia-candidate","messages":[{"role":"user","content":"\\u4f60\\u597d"}],"reasoning_effort":"h\\u0069gh"}'
        after=adapt_request_body(before,candidate_model_id=MODEL)
        self.assertEqual(after.body,before.replace(b'"h\\u0069gh"',b'"xhigh"'))

    def test_ambiguous_or_invalid_requests_rejected(self):
        bodies=[b'{',b'[]',b'\xff',b'{"model":"qwen38-27b-nvidia-candidate","reasoning_effort":"high","reasoning_effort":"low"}',b'{"model":"qwen38-27b-nvidia-candidate","chat_template_kwargs":{"reasoning_effort":"high","reasoning_effort":"low"}}']
        for body in bodies:
            with self.assertRaises(InvalidRequestBody):adapt_request_body(body,candidate_model_id=MODEL)

    def test_roles_are_not_rewritten(self):
        for history in [[dict(role='developer',content='Instruction'),dict(role='user',content='Question')],[dict(role='system',content='First'),dict(role='system',content='Second'),dict(role='user',content='Question')]]:
            before=encode(messages=history,reasoning_effort='high')
            after=adapt_request_body(before,candidate_model_id=MODEL)
            self.assertEqual(json.loads(after.body)['messages'],history)
            with self.assertRaises(jinja2.TemplateError):render(after.body)


if __name__=='__main__':unittest.main(verbosity=2)
