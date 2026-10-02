from common import *
import shutil,ast,difflib
parent=NV/'production-dense-grouped-20260922';dest=NV/'candidate-v43-renderer-overlap-20260923'
assert not dest.exists();m=read(parent/'MANIFEST.json');dest.mkdir()
for rel,e in m['files'].items():
 assert sha(parent/rel)==e['sha256'];p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/rel,p)
changes={}
def put(rel,before,after):
 assert before!=after;ast.parse(after);p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(after);changes[rel]=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='before/'+rel,tofile='after/'+rel));m['files'][rel]={}
# Keep multimodal warmup's exact operations and error/cache cleanup behavior.
rel='overlay/vllm/renderers/base.py';s=(Q/'recovery/source-candidate/image-vllm-renderers-base.py').read_text();b=s
start=s.index('            if self.mm_processor:',s.index('    def warmup(self,'));end=s.index('\n    async def clear_mm_cache_async',start)
body=s[start:end]
replacement='''            if getattr(self, "_startup_mm_warmed", False):
                # Consume the earlier warmup once. Later explicit warmup calls
                # keep upstream behavior and run again.
                self._startup_mm_warmed = False
            else:
                self._warmup_multimodal_processors()

    def _warmup_multimodal_processors(self) -> None:
        with set_default_torch_num_threads(1):
'''+body+'''
    def prewarm_multimodal_for_startup(self) -> None:
        """CPU renderer preparation while an independent EngineCore starts.

        Called synchronously in the API process, after EngineCore spawn and
        before its readiness wait. No background thread or persistent worker.
        Normal API admission still waits for both paths to complete.
        """
        self._warmup_multimodal_processors()
        self._startup_mm_warmed = True
'''
s=s[:start]+replacement+s[end:];put(rel,b,s)
# Explicit callback argument, never attached to or serialized in VllmConfig.
rel='overlay/vllm/v1/engine/core_client.py';s=(Q/'recovery/source-candidate/image-vllm-v1-engine-core_client.py').read_text();b=s
start=s.index('    def make_async_mp_client(');end=s.index('\n    @abstractmethod',start);part=s[start:end]
part=part.replace('        client_index: int = 0,','        client_index: int = 0,\n        startup_callback: Callable[[], None] | None = None,')
part=part.replace('return AsyncMPClient(*client_args)','return AsyncMPClient(*client_args, startup_callback=startup_callback)');s=s[:start]+part+s[end:]
start=s.index('class MPClient(');end=s.index('        self.vllm_config = vllm_config',start)
part=s[start:end].replace('        client_addresses: dict[str, Any] | None = None,','        client_addresses: dict[str, Any] | None = None,\n        startup_callback: Callable[[], None] | None = None,');s=s[:start]+part+s[end:]
needle='                    tensor_queue = engine_launch.tensor_queue\n';assert s.count(needle)==1
s=s.replace(needle,needle+'''                    # CPU-only work runs on the API main thread while the
                    # separately spawned engine initializes. Exiting this
                    # context still waits for complete engine readiness.
                    if startup_callback is not None:
                        startup_callback()
''')
start=s.index('class AsyncMPClient(');end=s.index('        self.client_count = client_count',start);part=s[start:end]
part=part.replace('        client_index: int = 0,','        client_index: int = 0,\n        startup_callback: Callable[[], None] | None = None,')
part=part.replace('            client_addresses=client_addresses,','            client_addresses=client_addresses,\n            startup_callback=startup_callback,');s=s[:start]+part+s[end:];put(rel,b,s)
rel='overlay/vllm/v1/engine/async_llm.py';s=(parent/rel).read_text();b=s
needle='''            client_index=client_index,
        )

        # Loggers.''';assert s.count(needle)==1
s=s.replace(needle,'''            client_index=client_index,
            startup_callback=(
                renderer.prewarm_multimodal_for_startup
                if self.vllm_config.parallel_config.data_parallel_size == 1
                and not client_addresses
                else None
            ),
        )

        # Loggers.''');put(rel,b,s)
rows=read(dest/'overlay-integrity.json')
for rel in changes:
 if not any(row['path']==rel for row in rows):rows.append({'path':rel,'target':'/usr/local/lib/python3.12/dist-packages/'+rel.removeprefix('overlay/')})
for row in rows:row.update(bytes=(dest/row['path']).stat().st_size,sha256=sha(dest/row['path']))
save(dest/'overlay-integrity.json',rows)
m.update(state='SEALED_CANDIDATE_NOT_PROMOTED',parent_manifest_sha256=sha(parent/'MANIFEST.json'),qualification_variable='renderer-overlap',diagnostic_only=False)
m['files']={rel:{'bytes':(dest/rel).stat().st_size,'sha256':sha(dest/rel)} for rel in m['files']};save(dest/'MANIFEST.json',m)
module('renderer_sealed',dest/'launcher.py').static()
out=Q/'recovery/evidence/renderer-overlap-assembly';out.mkdir()
(out/'source.patch').write_text('\n'.join(changes.values()))
save(out/'receipt.json',{'state':'SEALED_CANDIDATE_NOT_PROMOTED','parent':str(parent),'parent_manifest':sha(parent/'MANIFEST.json'),'candidate':str(dest),'manifest':sha(dest/'MANIFEST.json'),'variable':'Move existing CPU multimodal processor warmup into independent EngineCore startup wait; retain chat warmup, all cleanup, admission and GPU initialization gates.','files':{rel:sha(dest/rel) for rel in changes},'patch_sha256':sha(out/'source.patch')})
print(dest)
