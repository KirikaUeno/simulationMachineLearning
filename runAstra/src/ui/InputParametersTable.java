package ui;

import data.InputParameter;

import javax.swing.table.DefaultTableModel;
import java.util.List;

public class InputParametersTable extends DefaultTableModel {

    public List<InputParameter> auList;

    public InputParametersTable(List<InputParameter> _auList) {
        this.auList = _auList;
    }

    public int getRowCount() {
        if (auList == null) {
            return 0;
        }
        return auList.size();
    }

    public int getColumnCount() {
        return 2;
    }

    public String getColumnName(int columnIndex) {
        if (columnIndex == 0) {
            return "parameter";
        } else if (columnIndex == 1) {
            return "value";
        }
        return null;
    }

    public Class<?> getColumnClass(int columnIndex) {
        return Object.class;
    }

    public boolean isCellEditable(int rowIndex, int columnIndex) {
        return columnIndex == 1;
    }

    public Object getValueAt(int rowIndex, int columnIndex) {
        InputParameter au = auList.get(rowIndex);

        if (columnIndex == 0) {
            return au.name;
        } else if (columnIndex == 1) {
            return au.value;
        }
        return "";
    }

    public void setValueAt(Object obj, int rowIndex, int columnIndex) {
        InputParameter au = auList.get(rowIndex);

        if (columnIndex == 1) {
            try {
                au.value = Double.parseDouble((String)obj);
            } catch (NumberFormatException e) {
                e.printStackTrace();
            }
        }
    }
}
