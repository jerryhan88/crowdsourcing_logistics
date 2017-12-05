from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import sys, platform


def cythonize(fileName):
    ext_modules = [
        Extension('%s' % fileName,
                  ['%s.pyx' % fileName], include_dirs=['.']),
    ]
    setup(
        name='crowdsourcing_logistics',
        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules,
        script_args=['build_ext'],
        options={'build_ext': {'inplace': True, 'force': True}}
    )
    print('******** CYTHON COMPLETE ******')


def terminal_exec():
    plf = platform.platform()
    if plf.startswith('Linux'):
        # Linux server
        args = sys.argv
        if len(args) == len(['pyFile', 'fileName']):
            _, fileName = args
            cythonize(fileName)
        else:
            print('******** Error ******')
            print('******** Type packageName and fileName ******')
    elif plf.startswith('Darwin'):
        # Mac
        fileName = 'gh_mBundling'
        cythonize(fileName)
    else:
        # Window ?
        pass


if __name__ == '__main__':
    cythonize('gh_mBundling')