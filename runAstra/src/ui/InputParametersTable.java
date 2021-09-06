package ui;

import data.InputParameter;

import javax.swing.table.DefaultTableModel;
import java.util.List;

public class InputParametersTable extends DefaultTableModel {

    public List<InputParameter> auList;
    public int firstPlotInd = 0;
    public int secondPlotInd = 1;

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
        return 3;
    }

    public String getColumnName(int columnIndex) {
        if (columnIndex == 0) {
            return "parameter";
        } else if (columnIndex == 1) {
            return "value";
        } else if (columnIndex == 2) {
            return "plot?";
        }
        return null;
    }

    public Class<?> getColumnClass(int columnIndex) {
        if(columnIndex == 2) return Boolean.class;
        return Object.class;
    }

    public boolean isCellEditable(int rowIndex, int columnIndex) {
        return columnIndex != 0;
    }

    public Object getValueAt(int rowIndex, int columnIndex) {
        InputParameter au = auList.get(rowIndex);

        if (columnIndex == 0) {
            return au.name;
        } else if (columnIndex == 1) {
            return au.value;
        } else if (columnIndex == 2) {
            return au.isHorizontalAxis;
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
        if (columnIndex == 2) {
            if(rowIndex>firstPlotInd && rowIndex>secondPlotInd){
                auList.get(secondPlotInd).isHorizontalAxis = false;
                secondPlotInd = rowIndex;
            } else if (rowIndex>firstPlotInd && rowIndex<secondPlotInd){
                auList.get(firstPlotInd).isHorizontalAxis = false;
                firstPlotInd = rowIndex;
            } else if (rowIndex<firstPlotInd && rowIndex <secondPlotInd){
                auList.get(secondPlotInd).isHorizontalAxis = false;
                secondPlotInd = firstPlotInd;
                firstPlotInd = rowIndex;
            }
            au.isHorizontalAxis = true;
            this.fireTableRowsUpdated(0,auList.size()-1);
        }
    }
}
