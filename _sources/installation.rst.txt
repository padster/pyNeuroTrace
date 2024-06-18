Installation
============

pyNeuroTrace can be installed with pip:

.. code-block:: shell

   pip install pyNeuroTrace


GPU Supported functions use the Python Library Cupy. This library has the following requirements:
* NVIDIA CUDA GPU with the Compute Capability 3.0 or larger.

* CUDA Toolkit: v11.2 / v11.3 / v11.4 / v11.5 / v11.6 / v11.7 / v11.8 / v12.0 / v12.1 / v12.2 / v12.3 / v12.4


'pyNeuroTrace' can be installed with Cupy using pip:

.. code-block:: shell
   
   pip install pyNeuroTrace[GPU]

If Cupy fails to build from the wheel using this command try installing Cupy first using a wheel that matches your CUDA Toolkit version ie:

.. code-block:: shell

   pip install cupy-cuda12x


Followed by this command to install pyNeuroTrace

.. code-block:: shell
   
   pip install pyNeuroTrace


For more information in on installing Cupy see the installation documentation [HERE](https://docs.cupy.dev/en/stable/install.html).