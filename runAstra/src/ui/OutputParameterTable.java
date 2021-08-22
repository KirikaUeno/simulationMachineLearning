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
        return 3;
    }

    public String getColumnName(int columnIndex) {
        switch (columnIndex){
            case 0:
                return "estimate?";
            case 1:
                return "parameter";
            case 2:
                return "prediction";
            default:
                return null;
        }
    }

    public Class<?> getColumnClass(int columnIndex) {
        return (columnIndex==0)?Boolean.class:Object.class;
    }

    public boolean isCellEditable(int rowIndex, int columnIndex) {
        return columnIndex == 0;
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
            default:
                return "";
        }
    }

    public void setValueAt(Object obj, int rowIndex, int columnIndex) {
        OutputParameter au = auList.get(rowIndex);

        if (columnIndex == 0) {
            au.estimate = (boolean)obj;
        }
    }
}
