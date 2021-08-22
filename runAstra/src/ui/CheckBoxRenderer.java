package ui;

import javax.swing.*;
import javax.swing.table.TableCellRenderer;
import java.awt.*;

public class CheckBoxRenderer implements TableCellRenderer {
    public final JCheckBox checkBox = new JCheckBox();

    public CheckBoxRenderer() {
        checkBox.setHorizontalAlignment(SwingConstants.CENTER);
    }

    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
        if (value instanceof Boolean) {
            boolean isOn = (Boolean) value;
            checkBox.setSelected(isOn);
        }

        Color c = Color.LIGHT_GRAY;
        if (isSelected) {
            checkBox.setBackground(c.darker());
        } else {
            checkBox.setBackground(c);
        }
        return checkBox;
    }
}
