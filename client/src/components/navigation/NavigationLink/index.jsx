import React from 'react';

import { NavLink } from 'react-router-dom';

import styles from './index.module.css';

const navigationLink = props => {
    return (
        <li className={styles.NavigationLink}>
            <NavLink to={props.to} activeClassName={styles.active} exact>
                {props.children}
            </NavLink>
        </li>
    );
};

export default navigationLink;