Installation
============

Core Package
-----------

.. code-block:: bash

   pip install quantumviz

With Dashboard (Web Interface)
-------------------------------

.. code-block:: bash

   pip install quantumviz[dashboard]

With Quantum Hardware Support
-----------------------------

.. code-block:: bash

   pip install quantumviz[all]

Development Install
-------------------

.. code-block:: bash

   git clone https://github.com/yourusername/quantumviz.git
   cd quantumviz
   pip install -e ".[dev,all]"

Requirements
------------

- Python 3.9+
- numpy >= 1.20
- matplotlib >= 3.5

Optional Dependencies
---------------------

- fastapi, uvicorn, pydantic (dashboard)
- qiskit, qiskit-ibm-runtime (hardware)
- pytest (testing)
