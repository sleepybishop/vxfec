"""Exercise Clos setup and cleanup without changing host networking."""
from pathlib import Path
import subprocess
import tempfile
import unittest


class RdmaRestoreTest(unittest.TestCase):
    def setup_section(self, output):
        preamble = '''set -eu
ORIG_RDMA_MODE=""
modprobe() { :; }
rdma() {
    if [[ "$*" == 'system show' ]]; then printf '%s\\n' "$1_OUTPUT"; else printf 'changed %s\\n' "$*"; fi
}
'''.replace('"$1_OUTPUT"', '"$RDMA_TEST_OUTPUT"')
        return subprocess.run(['bash', '-c', preamble + launcher_functions('enable_rxe') + '\nenable_rxe\nprintf "saved=%s\\n" "$ORIG_RDMA_MODE"'],
                              env={'PATH': '/usr/bin:/bin', 'RDMA_TEST_OUTPUT': output}, text=True, capture_output=True)

    def test_mode_is_independent_of_following_properties(self):
        for mode in ('shared', 'exclusive'):
            for suffix in ('', ' privileged-qkey off monitor on copy-on-fork on'):
                result = self.setup_section('netns ' + mode + suffix)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('saved=' + ('shared' if mode == 'shared' else '') + '\n', result.stdout)
                self.assertEqual('changed system set netns exclusive' in result.stdout, mode == 'shared')

    def test_unknown_format_stops_before_mutating_mode(self):
        result = self.setup_section('copy-on-fork on')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('changed', result.stdout)



def launcher_functions(*names):
    import re
    source = Path(__file__).with_name('run_rocev2_clos.sh').read_text()
    return '\n'.join(re.search(r'^' + name + r'\(\) \{.*?^\}', source, re.M | re.S).group() for name in names)


class NamespaceOwnershipTest(unittest.TestCase):
    def test_failed_setup_preserves_other_namespaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = '''set -eu
cd "$1"
WORK_TMP="$(mktemp -d)"
ORIG_RDMA_MODE=""
BG_PIDS=()
ALL_NS=(first second third)
created_namespaces=()
ip() {
    case "$1 $2" in
        'netns add') mkdir "$3" ;;
        'netns del') rmdir "$3" ;;
        '-n first') return 0 ;;
        *) return 1 ;;
    esac
}
''' + launcher_functions('cleanup', 'create_namespaces') + '''
trap cleanup EXIT
create_namespaces
'''
            for occupied in ('first', 'second', 'third'):
                with self.subTest(occupied=occupied):
                    (root / occupied).mkdir()
                    result = subprocess.run(['bash', '-c', script, 'namespace-test', directory], capture_output=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(list(root.iterdir()), [root / occupied])
                    (root / occupied).rmdir()


class OffloadConfigurationTest(unittest.TestCase):
    def invoke(self, features, command_status=0):
        prefix = '''set -eu
ip() {
    shift 3
    if [[ "$1 $2" == 'ethtool -K' ]]; then return "$CLOS_TEST_SET_STATUS"; fi
    [[ "$1 $2" == 'ethtool -k' ]]
    printf '%s\\n' "$CLOS_TEST_FEATURES"
}
'''
        return subprocess.run(['bash', '-c', prefix + launcher_functions('disable_link_offloads') + '\ndisable_link_offloads test eth0'],
                              env={'PATH': '/usr/bin:/bin', 'CLOS_TEST_FEATURES': features, 'CLOS_TEST_SET_STATUS': str(command_status)},
                              capture_output=True, text=True)

    def test_verified_off_features_allow_unsupported_optional_settings(self):
        fields = ('tcp-segmentation-offload', 'generic-segmentation-offload', 'generic-receive-offload',
                  'tx-udp-segmentation', 'rx-udp-gro-forwarding')
        features = '\n'.join(key + ': off [fixed]' for key in fields)
        self.assertEqual(self.invoke(features, 1).returncode, 0)
        for key in fields:
            with self.subTest(key=key):
                self.assertNotEqual(self.invoke(features.replace(key + ': off', key + ': on')).returncode, 0)

    def test_missing_core_feature_fails_verification(self):
        features = 'tcp-segmentation-offload: off\ngeneric-receive-offload: off\n'
        self.assertNotEqual(self.invoke(features).returncode, 0)


class ForwardingHealthTest(unittest.TestCase):
    def test_invalid_switch_stops_waiting_ranks(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            script = '''set -eu
WORK_TMP="$1"
sleep 10 & spine=$!
sleep 10 & rank=$!
trap 'kill "$spine" "$rank" 2>/dev/null || true; wait 2>/dev/null || true' EXIT
NUM_SPINES=1
SPINE_PIDS=("$spine")
printf 'stat_seq=1 valid=0\n' > "$WORK_TMP/spine0.stats"
''' + launcher_functions('read_stat_val', 'check_spine_health', 'monitor_case') + '''
if monitor_case "$rank"; then exit 2; fi
if wait "$rank"; then exit 3; fi
'''
            result = subprocess.run(['bash', '-c', script, 'health-test', directory], capture_output=True, text=True, timeout=3)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('invalid packet forwarding', result.stderr)

    def test_snapshot_waits_for_fresh_valid_statistics(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            script = '''set -eu
WORK_TMP="$1"
printf 'stat_seq=1 valid=1\n' > "$WORK_TMP/spine0.stats"
(
    trap 'printf "stat_seq=2 valid=1\\n" > "$WORK_TMP/next"; mv "$WORK_TMP/next" "$WORK_TMP/spine0.stats"' USR1
    touch "$WORK_TMP/ready"
    while :; do sleep 0.01; done
) & spine=$!
trap 'kill "$spine" 2>/dev/null || true; wait 2>/dev/null || true' EXIT
NUM_SPINES=1
SPINE_PIDS=("$spine")
while [[ ! -f "$WORK_TMP/ready" ]]; do sleep 0.005; done
''' + launcher_functions('read_stat_val', 'check_spine_health', 'flush_spine_stats') + '''
flush_spine_stats
[[ $(read_stat_val "$WORK_TMP/spine0.stats" stat_seq) == 2 ]]
'''
            result = subprocess.run(['bash', '-c', script, 'snapshot-test', directory], capture_output=True, text=True, timeout=8)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
