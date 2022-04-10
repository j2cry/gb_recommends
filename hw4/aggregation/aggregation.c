// build with
// gcc -I/usr/include/python3.9 -fPIC -shared -o libagg.so aggregation.c
#define PY_SSIZE_T_CLEAN
#include <Python.h>



int valueIsIn(PyObject* value, PyObject* array) {
    int exist = 0;
    Py_ssize_t size = PyList_GET_SIZE(array);
    for (int i = 0; i < size; i++) {
        exist = PyObject_RichCompareBool(value, PyList_GetItem(array, i), Py_EQ);
        if (exist)
            return exist;
    }
    return 0;
}


/*
 *      FUNC DEFINITIONS
 */
PyObject * uniquesFromColumns(PyObject *array) {
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* uniques = PyList_New(0);
    Py_ssize_t rowCount = PyList_GET_SIZE(array);

    int curCol = 0;
    Py_ssize_t maxCols = -1;
    while (curCol != maxCols) {
        for (int row = 0; row < rowCount; row++) {
            PyObject* rowData = PyList_GET_ITEM(array, row);
            Py_ssize_t colCount = PyList_GET_SIZE(rowData);
            if (maxCols < colCount) {
                maxCols = colCount;
            }

            for (int col = curCol; col < colCount; col++) {
                PyObject* value = PyList_GET_ITEM(rowData, col);
                // long val = PyLong_AsLong(value);
                // printf("processing value at [row=%d, col=%d, val=%ld]\n", row, col, val);
                if (valueIsIn(value, uniques)) continue;
                PyList_Append(uniques, value);
                break;
            }
        }
        curCol++;
    }
    PyGILState_Release(gstate);
    return uniques;
}

// PyList_New segfault: https://stackoverflow.com/questions/35774011/segment-fault-when-creating-pylist-new-in-python-c-extention
