from setuptools import find_packages, setup

package_name = "rocycle_robot"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="rocycle team",
    maintainer_email="student@example.com",
    description="Conveyor-based recycling sorting cobot — perception, tracking, policy, state machine",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "tracking_node = rocycle_robot.tracking_node:main",
        ],
    },
)
