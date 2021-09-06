package ui;

import data.OutputParameter;

import javax.swing.table.DefaultTableModel;
import java.util.List;

public class OutputParameterTable extends DefaultTableModel {

    public List<OutputParameter> auList;

    public OutputParameterTable(List<OutputParameter> _auList) {
        this.auList = _auList;
    }

    public int getRowCount() {
        if (auList == null) {
            return 0;
        }
        return auList.size();
    }

    public int getColumnCount() {
        return 4;
    }

    public String getColumnName(int columnIndex) {
        switch (columnIndex){
            case 0:
                return "estimate?";
            case 1:
                return "parameter";
            case 2:
                return "prediction";
            case 3:
                return "plot?";
            default:
                return null;
        }
    }

    public Class<?> getColumnClass(int columnIndex) {
        return (columnIndex==0 || columnIndex==3)?Boolean.class:Object.class;
    }

    public boolean isCellEditable(int rowIndex, int columnIndex) {
        return (columnIndex == 0 || columnIndex == 3);
    }

    public Object getValueAt(int rowIndex, int columnIndex) {
        OutputParameter au = auList.get(rowIndex);

        switch (columnIndex){
            case 0:
                return au.estimate;
            case 1:
                return au.name;
            case 2:
                return au.estimatedValue;
            case 3:
                return au.isZAxis;
            default:
                return "";
        }
    }

    public void setValueAt(Object obj, int rowIndex, int columnIndex) {
        OutputParameter au = auList.get(rowIndex);

        if (columnIndex == 0) {
            au.estimate = (boolean)obj;
        }
        if (columnIndex == 3) {
            for (OutputParameter p : auList) {
                p.isZAxis = false;
            }
            au.isZAxis = true;
        }
        this.fireTableRowsUpdated(0,auList.size()-1);
    }
}
