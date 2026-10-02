import unittest
from unittest.mock import patch
import startup_campaign_r2 as s

class StartupSamplerTest(unittest.TestCase):
    def test_precreate_intent_with_null_container_does_not_spawn_docker(self):
        sampler=s.StartupSampler.__new__(s.StartupSampler);sampler.candidate=s.BASE
        with patch.object(s,'active',return_value={'container_id':None,'pid':None,'instance_id':'pending'}),patch.object(s,'cmd') as command,patch.object(s.sampler.Sampler,'sample',return_value={'cpu':{}}):
            row=sampler.sample()
        command.assert_not_called();self.assertEqual(row['owner']['instance_id'],'pending');self.assertEqual(row['io'],{})

if __name__=='__main__':unittest.main()
