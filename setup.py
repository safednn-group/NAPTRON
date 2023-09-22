from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


if __name__ == '__main__':
    setup(
        name='safednn_naptron',
        version='0.0.1',
        description='SafeDNN package',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='SafeDNN team',
        packages=find_packages(),
        include_package_data=True,
        license='Apache License 2.0',
        install_requires=["mmdet>=2.20"],
    )
